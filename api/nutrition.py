import os
import re
from qdrant_client import QdrantClient
from openai import OpenAI
from api.embedding_utils import get_embedding_model
from api.logging_utils import setup_logger

logger = setup_logger(__name__)

COLLECTION_NAME = "nutrition_data"

async def get_nutrition_recommendation(query: str, limit: int = 3):
    """
    Get nutrition recommendations for pregnant women based on their details
    and generate a weekly diet plan tailored to their geographical region
    """
    logger.info(f"Getting nutrition recommendation for query: '{query}', limit: {limit}")
    
    try:
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        llama_api_key = os.getenv("LLAMA_API_KEY")
        e2e_networks_url = os.getenv("E2E_NETWORKS_URL")
        
        if not qdrant_url or not qdrant_api_key or not llama_api_key or not e2e_networks_url:
            logger.error("Missing required environment variables")
            raise ValueError("Missing required environment variables")
        
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        
        collections = client.get_collections().collections
        collection_exists = any(collection.name == COLLECTION_NAME for collection in collections)
        
        if not collection_exists:
            logger.error(f"Collection {COLLECTION_NAME} does not exist")
            raise ValueError(f"Collection {COLLECTION_NAME} does not exist")
        
        model = get_embedding_model()
        query_vector = model.encode([query])[0]
        
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=limit
        )
        
        nutrition_info = [res.payload.get("text", "") for res in results]
        combined_info = "\n\n".join(nutrition_info)
        
        region_keywords = ["region", "state", "city", "district", "area", "from", "lives in"]
        region = "unknown"
        
        for keyword in region_keywords:
            if keyword in query.lower():
                parts = query.lower().split(keyword)
                if len(parts) > 1:
                    region_part = parts[1].strip()
                    region_words = region_part.split()
                    if len(region_words) > 0:
                        region = region_words[0]
                        if len(region_words) > 1:
                            region += " " + region_words[1]
                        break
        
        prompt = f"""
You are a nutrition expert specializing in maternal health. Based on the following information about a pregnant woman and nutritional guidelines, create a personalized 7-day diet plan.

Woman's details: {query}

Relevant nutrition information: {combined_info}

Geographic region: {region}

Create a detailed 7-day diet plan specifically tailored for this pregnant woman considering her geographical region, local food availability, and nutritional needs. Include breakfast, lunch, dinner, and snacks for each day. Focus on providing adequate protein, iron, folate, calcium, and other essential nutrients for pregnancy.

Your response should be in a structured JSON format with the following structure:
{{
  "day1": {{
    "breakfast": "Detailed breakfast description",
    "morning_snack": "Detailed morning snack description",
    "lunch": "Detailed lunch description",
    "evening_snack": "Detailed evening snack description",
    "dinner": "Detailed dinner description"
  }},
  ... and so on for all 7 days
}}

Make sure your response is valid JSON that can be parsed directly.
"""
        
        llama_client = OpenAI(
            base_url=e2e_networks_url,
            api_key=llama_api_key
        )
        
        try:
            completion = llama_client.chat.completions.create(
                model='llama3_2_3b_instruct',
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=1024,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=1
            )
            
            diet_plan_text = completion.choices[0].message.content
            
            json_pattern = r'```json\s*(.*?)\s*```|```\s*(.*?)\s*```|(\{.*\})'
            json_match = re.search(json_pattern, diet_plan_text, re.DOTALL)
            
            if json_match:
                for group in json_match.groups():
                    if group:
                        diet_plan_text = group
                        break
            
            diet_plan_text = diet_plan_text.strip()
            if not diet_plan_text.startswith('{'):
                start_idx = diet_plan_text.find('{')
                if start_idx != -1:
                    diet_plan_text = diet_plan_text[start_idx:]
            
            if not diet_plan_text.endswith('}'):
                end_idx = diet_plan_text.rfind('}')
                if end_idx != -1:
                    diet_plan_text = diet_plan_text[:end_idx+1]
            
            import json
            try:
                diet_plan_json = json.loads(diet_plan_text)
                
                structured_response = {}
                
                for day_num in range(1, 8):
                    day_key = f"day{day_num}"
                    
                    if day_key not in diet_plan_json:
                        structured_response[day_key] = {
                            "breakfast": "Not specified",
                            "morning_snack": "Not specified",
                            "lunch": "Not specified",
                            "evening_snack": "Not specified",
                            "dinner": "Not specified"
                        }
                    else:
                        day_data = diet_plan_json[day_key]
                        structured_response[day_key] = {
                            "breakfast": day_data.get("breakfast", "Not specified"),
                            "morning_snack": day_data.get("morning_snack", "Not specified"),
                            "lunch": day_data.get("lunch", "Not specified"),
                            "evening_snack": day_data.get("evening_snack", "Not specified"),
                            "dinner": day_data.get("dinner", "Not specified")
                        }
                
                return {
                    "diet_plan": structured_response,
                    "region": region
                }
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse diet plan as JSON: {str(e)}")
                
                structured_response = {}
                
                day_patterns = [
                    r'Day (\d+)[:\s]*(.*?)(?=Day \d+|$)',
                    r'day(\d+)[:\s]*(.*?)(?=day\d+|$)',
                    r'DAY (\d+)[:\s]*(.*?)(?=DAY \d+|$)'
                ]
                
                for pattern in day_patterns:
                    day_matches = re.findall(pattern, diet_plan_text, re.DOTALL | re.IGNORECASE)
                    if day_matches:
                        for day_num, day_content in day_matches:
                            day_key = f"day{day_num}"
                            day_data = {
                                "breakfast": "Not specified",
                                "morning_snack": "Not specified",
                                "lunch": "Not specified",
                                "evening_snack": "Not specified",
                                "dinner": "Not specified"
                            }
                            
                            meal_patterns = {
                                "breakfast": r'breakfast[:\s]*(.*?)(?=lunch|dinner|snack|$)',
                                "morning_snack": r'morning snack[:\s]*(.*?)(?=lunch|dinner|breakfast|evening|$)',
                                "lunch": r'lunch[:\s]*(.*?)(?=breakfast|dinner|snack|$)',
                                "evening_snack": r'evening snack[:\s]*(.*?)(?=breakfast|lunch|dinner|morning|$)',
                                "dinner": r'dinner[:\s]*(.*?)(?=breakfast|lunch|snack|$)'
                            }
                            
                            for meal_key, meal_pattern in meal_patterns.items():
                                meal_match = re.search(meal_pattern, day_content, re.DOTALL | re.IGNORECASE)
                                if meal_match:
                                    day_data[meal_key] = meal_match.group(1).strip()
                            
                            structured_response[day_key] = day_data
                
                if not structured_response:
                    for day_num in range(1, 8):
                        structured_response[f"day{day_num}"] = {
                            "breakfast": "Not specified",
                            "morning_snack": "Not specified",
                            "lunch": "Not specified",
                            "evening_snack": "Not specified",
                            "dinner": "Not specified"
                        }
                
                return {
                    "diet_plan": structured_response,
                    "region": region,
                    "note": "The diet plan could not be parsed as JSON and was manually structured."
                }
                
        except Exception as e:
            logger.error(f"Error generating diet recommendation: {str(e)}")
            raise
    
    except Exception as e:
        logger.error(f"Error in get_nutrition_recommendation: {str(e)}")
        raise 