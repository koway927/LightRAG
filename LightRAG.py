import os
import requests
import json
import re
from typing import Dict, List, Any, Tuple
import numpy as np
import nano_vectordb as nano
from google.genai import types
import PyPDF2
from google import genai
import argparse
from sentence_transformers import SentenceTransformer
class LightRAG:

    def __init__(self, api_key):
        
        self.client = genai.Client(api_key=api_key)
        self.api_key = api_key
        self.id_counter = 0
        
        self.dim = 384
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.entities_vdb = nano.NanoVectorDB(self.dim , storage_file="entities.json")
        self.relationships_vdb = nano.NanoVectorDB(self.dim , storage_file="relationships.json")
        self.knowledge_graph = {'nodes': {}, 'edges': {}}

    def _call_gemini(self, prompt):

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0) # Disables thinking
            ),
        )
        return response.candidates[0].content.parts[0].text
    
    def _get_embedding(self, text):

        embedding = self.embedding_model.encode([text])[0]
        return np.array(embedding)
        
    def _get_graph_construct_prompt(self, text_chunk):

        return f"""
            -Goal-
            Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

            -Steps-
            1. Identify all entities. For each identified entity, extract the following information:
            - entity_name: Name of the entity, capitalized
            - entity_type: One of the following types: [organization, person, geo, event, concept, technology]
            - entity_description: Comprehensive description of the entity's attributes and activities based on the text.
            Format each entity as "<entity><entity_name>...</entity_name><entity_type>...</entity_type><entity_description>...</entity_description></entity>"

            2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are "clearly related" to each other.
            For each pair of related entities, extract the following information:
            - source_entity: name of the source entity, as identified in step 1
            - target_entity: name of the target entity, as identified in step 1
            - relationship_description: explanation as to why you think the source entity and the target entity are related to each other, based on the text.
            - relationship_strength: a numeric score from 0.1 to 1.0 indicating strength of the relationship between the source entity and target entity
            - relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details.
            Format each relationship as "<relationship><source_entity>...</source_entity><target_entity>...</target_entity><relationship_description>...</relationship_description><relationship_strength>...</relationship_strength><relationship_keywords>...</relationship_keywords></relationship>"

            3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use "##" as the list delimiter.

            4. When finished, output <COMPLETE>

            -Real Data-
            Entity types: [organization, person, geo, event, concept, technology]
            Text: "{text_chunk}"

            Output:
            """

    def _parse_and_populate_graph(self, llm_output):

        def parse_block(block_content):
            obj = {}
            # Regex to find all <key>value</key> pairs inside the block
            pattern = r"<(\w+)>(.*?)<\/\1>"
            matches = re.findall(pattern, block_content, re.DOTALL)
            for key, value in matches:
                obj[key.strip()] = value.strip()
            return obj

        # Extract all <entity>...</entity> blocks
        entity_blocks = re.findall(r"<entity>(.*?)<\/entity>", llm_output, re.DOTALL)
        for block in entity_blocks:
            entity_data = parse_block(block)
            if 'entity_name' in entity_data:
                node_key = entity_data['entity_name'].lower()

                if node_key not in self.knowledge_graph['nodes'] and entity_data.get('entity_description'):
                    self.knowledge_graph['nodes'][node_key] = entity_data
                    entity_value_embedding = self._get_embedding(entity_data['entity_description'])
                    r = self.entities_vdb.upsert([{"__id__": self.id_counter, "__vector__": entity_value_embedding, "key": node_key}])
                    self.id_counter += 1

        # Extract all <relationship>...</relationship> blocks
        relationship_blocks = re.findall(r"<relationship>(.*?)<\/relationship>", llm_output, re.DOTALL)
        for block in relationship_blocks:
            rel_data = parse_block(block)
            if 'source_entity' in rel_data and 'target_entity' in rel_data:
                edge_key = f"{rel_data['source_entity'].lower()}|{rel_data['target_entity'].lower()}"

                if edge_key not in self.knowledge_graph['edges'] and rel_data.get('relationship_description'):
                    self.knowledge_graph['edges'][edge_key] = rel_data
                    edge_value_embedding = self._get_embedding(rel_data['relationship_description'])
                    r = self.relationships_vdb.upsert([{"__id__": self.id_counter, "__vector__": edge_value_embedding, "key": edge_key}])
                    self.id_counter += 1
        print("Document indexing complete.")

    def build_knowledge_graph(self, document_text):

        self.knowledge_graph = {'nodes': {}, 'edges': {}}
        
        chunks = [p.strip() for p in document_text.split('<PAGE_BREAK>') if p.strip()]
        if not chunks:
            chunks = [document_text]


        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i + 1}/{len(chunks)}...")
          
            prompt = self._get_graph_construct_prompt(chunk)
            llm_response = self._call_gemini(prompt)
            self._parse_and_populate_graph(llm_response)
            
        print("Knowledge graph construction complete.")


        

    def _vector_search(self, query_keywords, search_type, top_k = 5) :
        """
        Performs vector similarity search using nano-vectordb.
        """
        
        if not query_keywords:
            return []
        retrieved_items = []   
        vdb = self.entities_vdb if search_type == 'entity' else self.relationships_vdb
        target_items = self.knowledge_graph['nodes'] if search_type == 'entity' else self.knowledge_graph['edges']

        retrieved_items = []
        for keyword in query_keywords:

            embedding = self._get_embedding(keyword)
            search_results = vdb.query(embedding, top_k=top_k, better_than_threshold=0.01)
            

            for res in search_results:
                item_key = res['key']
                retrieved_items.append(target_items[item_key])
        return retrieved_items

    def _get_keyword_extraction_prompt(self, query):
        """
        Creates the prompt for keyword extraction based on Figure 6 of the paper.
        """
        return f"""
        -Role-
        You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query.

        -Goal-
        Given the query, list both high-level and low-level keywords. High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.

        -Instructions-
        Output the keywords in JSON format.
        The JSON should have two keys:
        - "high_level_keywords": an array of strings for overarching concepts or themes.
        - "low_level_keywords": an array of strings for specific entities or details.

        -Example-
        Query: "How does international trade influence global economic stability?"
        Output: {{ "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"], "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"] }}

        -Real Data-
        Query: "{query}"
        Output:
        """


    def answer_question(self, query):
        """
        Answers a question based on the constructed knowledge graph.

        Args:
            query: The user's question.

        Returns:
            A tuple containing (final_answer, retrieval_context_string).
        """
        if not self.knowledge_graph['nodes']:
            return "The knowledge graph has not been built yet. Please call `build_knowledge_graph` first.", ""

        # Keyword Extraction
        print("Extracting keywords from query...")
        keyword_prompt = self._get_keyword_extraction_prompt(query)
        try:
            keyword_response = self._call_gemini(keyword_prompt)
        
            cleaned_json_str = re.sub(r'```json\n?|```', '', keyword_response)
            keywords = json.loads(cleaned_json_str)
        except Exception as e:
            return f"Failed to extract keywords: {e}", ""

        # Dual-Level Retrieval
        print("Performing dual-level retrieval...")
        retrieved_entities = self._vector_search(keywords.get('low_level_keywords', []), 'entity')
        retrieved_relations = self._vector_search(keywords.get('high_level_keywords', []), 'relation')

        # Incorporate High-Order Relatedness (1-hop neighbors)
        context_elements = {}
        for entity in retrieved_entities:
            context_elements[entity['entity_name'].lower()] = {'type': 'entity', 'data': entity}
        
        for rel in retrieved_relations:
            edge_key = f"{rel['source_entity'].lower()}|{rel['target_entity'].lower()}"
            context_elements[edge_key] = {'type': 'relation', 'data': rel}

        # Expand with neighbors of retrieved entities
        for entity in retrieved_entities:
            entity_name_lower = entity['entity_name'].lower()
            for edge_key, edge_data in self.knowledge_graph['edges'].items():
                if entity_name_lower in edge_key.split('|'):
                    if edge_key not in context_elements:
                        context_elements[edge_key] = {'type': 'relation', 'data': edge_data}

        # Assemble Context
        print("Assembling context...")
        context_string = "### Retrieved Context from Knowledge Graph ###\n\n"
        if not context_elements:
            context_string += "No specific context was retrieved from the document.\n"
        else:
            for element in context_elements.values():
                if element['type'] == 'entity':
                    data = element['data']
                    context_string += f"[ENTITY] Name: {data.get('entity_name')}, Type: {data.get('entity_type')}, Description: {data.get('entity_description')}\n\n"
                elif element['type'] == 'relation':
                    data = element['data']
                    context_string += f"[RELATION] Between: {data.get('source_entity')} and {data.get('target_entity')}, Description: {data.get('relationship_description')}\n\n"

        # Answer Generation
        print("Generating final answer...")
        final_prompt = f"""
            Based on the following context, please provide a comprehensive answer to the user's query. Synthesize the information from the context into a coherent response. Do not just list the context items. If the context is empty or irrelevant, state that the document does not contain the necessary information to answer the question.

            {context_string}

            ### User Query ###
            {query}

            ### Answer ###
            """
        try:
            final_answer = self._call_gemini(final_prompt)
            return final_answer, context_string
        except Exception as e:
            return f"Failed to generate final answer: {e}", context_string


def main():
    """Main function to run the LightRAG command-line demo."""
    print("--- LightRAG Python Implementation ---")
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path", help="Path to the PDF file to read")
    args = parser.parse_args()
    pdf_path = args.pdf_path
    
    api_key = "AIzaSyBTa0GD3TLJH0Vfl5Us8vOc9tU4ICqRNro"

    rag_system = LightRAG(api_key)

    if not os.path.isfile(pdf_path):
        print("File not found. Exiting.")
        return
    print(f"Loading PDF from {pdf_path}...")
    document_text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    document_text += page_text + "<PAGE_BREAK>"
    except Exception as e:
        print(f"Failed to read PDF: {e}")
        return

    if not document_text.strip():
        print("No text could be extracted from the PDF. Exiting.")
        return
    
    print("\n--- Indexing ---")


    rag_system.build_knowledge_graph(document_text)
    

    print("\n---  Question Answering ---")
    print("You can now ask questions about the document. Type 'exit' or 'quit' to end.")

    while True:
        query = input("\nYour Question: ")
        if query.lower() in ['exit', 'quit']:
            break
        
        answer, context = rag_system.answer_question(query)
        
        print("\n--- Retrieval Context ---")
        print(context)
        print("\n--- Final Answer ---")
        print(answer)
        print("-" * 20)

if __name__ == "__main__":
    main()
