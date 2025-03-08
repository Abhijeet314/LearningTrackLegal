from flask import Flask, request, jsonify
import os
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any
from together import Together
import json
import re
import requests
from dotenv import load_dotenv
import random
from datetime import datetime
import hashlib
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load environment variables
load_dotenv()

# Define Pydantic models for structured output
class Resource(BaseModel):
    title: str = Field(..., description="Title of the resource")
    description: str = Field(..., description="Brief description of the resource")
    url: str = Field(..., description="URL to the resource (article, blog, video)")
    type: Literal["article", "video", "blog", "book"] = Field(..., description="Type of resource")
    time_estimate: str = Field(..., description="Estimated time to complete this resource")
    content_summary: Optional[str] = Field(None, description="Summary of resource content (populated after search)")

class QuizQuestion(BaseModel):
    question: str = Field(..., description="The question text")
    options: List[str] = Field(..., description="List of possible answers")
    correct_answer_index: int = Field(..., description="Index of the correct answer (0-based)")
    explanation: str = Field(..., description="Explanation of the correct answer")

class Module(BaseModel):
    title: str = Field(..., description="Title of the module")
    description: str = Field(..., description="Detailed description of what will be covered in this module")
    learning_objectives: List[str] = Field(..., description="List of learning objectives for this module")
    key_concepts: List[str] = Field(..., description="List of key concepts covered in this module")
    resources: List[Resource] = Field(..., description="List of resources for this module")
    practice_activities: List[str] = Field(..., description="Suggested practice activities for this module")
    quiz_questions: Optional[List[QuizQuestion]] = Field([], description="Quiz questions for this module")
    
class LearningTrack(BaseModel):
    track_title: str = Field(..., description="Title of the learning track")
    track_description: str = Field(..., description="Detailed description of the learning track")
    target_audience: str = Field(..., description="Description of who this track is designed for")
    prerequisites: List[str] = Field(..., description="List of prerequisites for this track")
    time_commitment: str = Field(..., description="Estimated time commitment for the entire track")
    modules: List[Module] = Field(..., description="List of modules in this track")

class UserProgress(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user")
    track_id: str = Field(..., description="ID of the learning track")
    completed_objectives: Dict[str, List[int]] = Field(default_factory=dict, description="Completed learning objectives by module")
    quiz_results: Dict[str, Dict[int, bool]] = Field(default_factory=dict, description="Quiz results by module and question")
    tokens_earned: int = Field(default=0, description="Number of tokens earned")
    last_updated: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Last update timestamp")

# Initialize Together AI client
def get_together_client():
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        return None
    return Together(api_key=api_key)

# Google Search API integration
def search_google(query, api_key=None, cx=None, result_type=None, num=3):
    """
    Search Google using Custom Search API
    
    Parameters:
    query (str): Search query
    api_key (str): Google API key
    cx (str): Google Custom Search Engine ID
    result_type (str): Type of result (article, video, blog, book)
    num (int): Number of results to return
    
    Returns:
    dict: Search results with title, link, and snippet
    """
    # Use environment variables if not provided
    api_key = api_key or os.getenv("GOOGLE_API_KEY") or "AIzaSyBBlilMKtxHIGr8-bfGSmNypvGU0CWoMBM"
    cx = cx or os.getenv("GOOGLE_CSE_ID") or "b24b95e96a98e4d81"
    
    if not api_key or not cx:
        return fallback_search_results(query, result_type)
    
    # Customize search based on result type
    search_params = {
        "key": api_key,
        "cx": cx,
        "q": query,
        "num": num
    }
    
    # Add specific search parameters based on resource type
    if result_type == "video":
        search_params["q"] += " video tutorial"
        search_params["fileType"] = "mp4"
    elif result_type == "article":
        search_params["q"] += " article"
    elif result_type == "blog":
        search_params["q"] += " blog"
    elif result_type == "book":
        search_params["q"] += " book"
    
    try:
        response = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params=search_params
        )
        
        if response.status_code == 200:
            results = response.json()
            return results.get("items", [])
        else:
            return fallback_search_results(query, result_type)
    except Exception as e:
        return fallback_search_results(query, result_type)

# Fallback search method when API is not available
def fallback_search_results(query, result_type):
    """Provide fallback search results when Google API is unavailable"""
    domains = {
        "article": ["law.cornell.edu", "lawreview.org", "americanbar.org", "findlaw.com"],
        "video": ["youtube.com/watch", "vimeo.com", "harvard.edu/video"],
        "blog": ["abovethelaw.com", "legalblogs.findlaw.com", "lawprofessorblogs.com"],
        "book": ["amazon.com/Law", "westacademic.com", "books.google.com"]
    }
    
    # Choose relevant domains for this result type
    result_domains = domains.get(result_type, domains["article"])
    
    # Generate a deterministic but seemingly random URL based on the query
    query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
    
    # Create a list of mock results
    results = []
    for i in range(3):
        domain = result_domains[i % len(result_domains)]
        slug = query.lower().replace(" ", "-")[:30]
        url = f"https://{domain}/{slug}-{query_hash}"
        
        results.append({
            "title": f"{query.title()} - {'Reference' if result_type=='article' else result_type.title()}",
            "link": url,
            "snippet": f"A comprehensive {result_type} about {query} in legal studies and practice."
        })
    
    return results

# Function to get LLM response for generating learning tracks
def get_llm_response(prompt, model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", temperature=0.7, max_tokens=4096):
    try:
        client = get_together_client()
        if not client:
            return "Error: Together AI API key not found"
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error querying Together AI: {e}"

# Create prompt template for generating learning tracks
def create_learning_track_prompt(category, specific_area):
    # Create a more explicit and detailed prompt with examples and emphasis on valid resource types
    prompt = f"""
    You are an expert legal educator tasked with creating a comprehensive learning track for a law student.
    
    User Category: {category}
    Specific Exam or Interest Area: {specific_area}
    
    Please create a detailed learning track with multiple modules. Each module should include:
    1. A clear title and description
    2. Learning objectives
    3. Key concepts
    4. Resource topics (not URLs - just the topics to search for)
    5. Practice activities
    
    The track should be tailored to the user's needs:
    - "For Fun" tracks should be accessible, engaging, and cover interesting legal topics.
    - "Exam Preparation" tracks should be structured around exam requirements, with strategic study plans.
    - "Deep Knowledge" tracks should be comprehensive and include advanced concepts and resources.
    
    IMPORTANT: For all resources, you MUST use ONLY these resource types: "article", "video", "blog", or "book". 
    Do NOT use any other values like "wiki", "website", etc. Always classify the resource as one of these four types.
    
    IMPORTANT: For URLs, DO NOT CREATE ACTUAL URLS. Instead, provide descriptive titles and topics that our system will use to search for relevant content.
    
    THE RESPONSE MUST BE A VALID JSON OBJECT WITH EXACTLY THE FOLLOWING STRUCTURE (This is critical):
    
    {{
      "track_title": "Title of the learning track",
      "track_description": "Detailed description of the learning track",
      "target_audience": "Description of who this track is designed for",
      "prerequisites": ["Prerequisite 1", "Prerequisite 2"],
      "time_commitment": "Estimated time commitment for the entire track",
      "modules": [
        {{
          "title": "Module title",
          "description": "Module description",
          "learning_objectives": ["Objective 1", "Objective 2"],
          "key_concepts": ["Concept 1", "Concept 2"],
          "resources": [
            {{
              "title": "Resource title",
              "description": "Resource description about a specific topic",
              "url": "resource topic to search for",
              "type": "article", 
              "time_estimate": "1 hour"
            }}
          ],
          "practice_activities": ["Activity 1", "Activity 2"]
        }}
      ]
    }}
    
    Remember, resource types MUST be one of: "article", "video", "blog", or "book".
    Your response should contain ONLY the JSON object with no additional text or explanation.
    """
    return prompt

# Function to generate quiz questions for a module
def generate_quiz_questions(module_data, num_questions=3):
    """Generate quiz questions based on module content using LLM"""
    
    # Create a comprehensive prompt with all module information
    resource_summaries = "\n".join([f"- {r.title}: {r.description}" for r in module_data.resources])
    key_concepts = "\n".join([f"- {c}" for c in module_data.key_concepts])
    
    quiz_prompt = f"""
    Create {num_questions} quiz questions based on the following module content:
    
    MODULE TITLE: {module_data.title}
    
    MODULE DESCRIPTION: {module_data.description}
    
    KEY CONCEPTS:
    {key_concepts}
    
    RESOURCE SUMMARIES:
    {resource_summaries}
    
    Each quiz question should have:
    1. A clear question testing understanding of key concepts
    2. Four multiple choice options (A, B, C, D)
    3. The index of the correct answer (0 for A, 1 for B, 2 for C, 3 for D)
    4. A brief explanation of why the answer is correct
    
    Output each question in valid JSON format like this example:
    [
      {{
        "question": "What is the primary purpose of the First Amendment?",
        "options": ["To protect property rights", "To ensure freedom of speech and religion", "To establish the right to bear arms", "To guarantee a fair trial"],
        "correct_answer_index": 1,
        "explanation": "The First Amendment primarily protects freedom of speech, religion, press, assembly, and petition."
      }},
      {{
        "question": "Another question here?",
        "options": ["Option A", "Option B", "Option C", "Option D"],
        "correct_answer_index": 2,
        "explanation": "Explanation of why Option C is correct."
      }}
    ]
    
    Return ONLY the JSON array with no additional text.
    """
    
    try:
        response = get_llm_response(quiz_prompt, temperature=0.3)
        
        # Extract JSON from response
        json_match = re.search(r'(\[.*\])', response, re.DOTALL)
        if json_match:
            quiz_json = json_match.group(1)
            questions = json.loads(quiz_json)
            
            # Validate the structure
            valid_questions = []
            for q in questions:
                if all(k in q for k in ["question", "options", "correct_answer_index", "explanation"]):
                    if isinstance(q["options"], list) and len(q["options"]) >= 2:
                        if isinstance(q["correct_answer_index"], int) and 0 <= q["correct_answer_index"] < len(q["options"]):
                            valid_questions.append(QuizQuestion(**q))
            
            return valid_questions
        else:
            # Fallback to create basic questions from key concepts
            return create_fallback_questions(module_data.key_concepts)
    except Exception as e:
        return create_fallback_questions(module_data.key_concepts)

# Create fallback quiz questions when LLM generation fails
def create_fallback_questions(key_concepts):
    """Create basic quiz questions based on key concepts"""
    questions = []
    
    # Use up to 3 key concepts to create questions
    for i, concept in enumerate(key_concepts[:3]):
        # Create a simple question format
        question = {
            "question": f"Which of the following best describes '{concept}'?",
            "options": [
                f"A definition related to {concept}",
                f"An example of {concept} in practice",
                f"A common misconception about {concept}",
                f"The historical development of {concept}"
            ],
            "correct_answer_index": 0,  # Default to first option
            "explanation": f"The first option provides the most accurate description of {concept}."
        }
        questions.append(QuizQuestion(**question))
    
    return questions

# Function to generate learning track with real resource links
def generate_learning_track(category, specific_area):
    try:
        prompt = create_learning_track_prompt(category, specific_area)
        
        # Get raw response from LLM
        response = get_llm_response(prompt)
        if not response or response.startswith("Error"):
            return {"error": response or "Failed to generate learning track"}
            
        # Extract and parse JSON from the response
        track_json = extract_json_from_response(response)
        
        if not track_json:
            return {"error": "Failed to extract valid JSON from the model response"}
            
        # Fix any invalid resource types before validation
        fixed_json = fix_resource_types(track_json)
        
        # Parse the response as a LearningTrack
        learning_track = LearningTrack.parse_raw(fixed_json)
        
        # Find real resources for each module
        enhanced_track = enhance_track_with_real_resources(learning_track)
        
        # Generate quiz questions for each module
        enhanced_track = add_quiz_questions_to_track(enhanced_track)
        
        return enhanced_track.dict()
    except Exception as e:
        return {"error": f"Error generating learning track: {str(e)}"}

# Add real resources to the track by searching
def enhance_track_with_real_resources(track):
    """Replace placeholder resource URLs with real ones from Google Search"""
    
    try:
        # Process each module
        for i, module in enumerate(track.modules):
            # Process each resource
            for j, resource in enumerate(module.resources):
                # Create search query from resource title and description
                search_query = f"legal {resource.title} {module.title} {resource.type}"
                
                # Get search results
                search_results = search_google(search_query, result_type=resource.type)
                
                # If we got search results, update the resource
                if search_results and len(search_results) > 0:
                    best_result = search_results[0]  # Take the top result
                    
                    # Update resource with real information
                    track.modules[i].resources[j].url = best_result.get("link")
                    track.modules[i].resources[j].content_summary = best_result.get("snippet", "")
                    
                    # If title is very generic, use search result title
                    if len(resource.title.split()) <= 3:
                        track.modules[i].resources[j].title = best_result.get("title", resource.title)
                
        return track
    except Exception as e:
        # Log error but continue with partial results
        print(f"Error enhancing resources: {e}")
        return track

# Add quiz questions to the track
def add_quiz_questions_to_track(track):
    """Generate and add quiz questions to each module"""
    
    try:
        # Process each module
        for i, module in enumerate(track.modules):
            # Generate quiz questions
            questions = generate_quiz_questions(module)
            
            # Add questions to module
            track.modules[i].quiz_questions = questions
        
        return track
    except Exception as e:
        # Log error but continue with partial results
        print(f"Error generating quiz questions: {e}")
        return track

# Helper function to fix resource types before validation
def fix_resource_types(json_str):
    try:
        # Parse JSON
        data = json.loads(json_str)
        
        # Fix resource types throughout all modules
        for module in data.get("modules", []):
            for resource in module.get("resources", []):
                # If resource type is not in allowed types, map it to a valid type
                if resource.get("type") not in ["article", "video", "blog", "book"]:
                    # Map invalid types to valid ones based on content or fallback to article
                    resource_title = resource.get("title", "").lower()
                    if "video" in resource_title or "tutorial" in resource_title:
                        resource["type"] = "video"
                    elif "blog" in resource_title:
                        resource["type"] = "blog"
                    elif "book" in resource_title:
                        resource["type"] = "book"
                    else:
                        # Default to article if can't determine
                        resource["type"] = "article"
        
        return json.dumps(data)
    except Exception as e:
        print(f"Error fixing resource types: {str(e)}")
        return json_str

# Helper function to extract JSON from LLM response
def extract_json_from_response(response):
    try:
        # Remove any markdown code block markers
        response = re.sub(r'```json\s*|\s*```', '', response)
        
        # Try to find JSON-like content in the response
        json_pattern = re.search(r'(\{.*\})', response, re.DOTALL)
        if json_pattern:
            json_str = json_pattern.group(1)
        else:
            json_str = response.strip()
        
        # Validate JSON
        parsed = json.loads(json_str)
        
        # Basic validation of required fields
        required_fields = ["track_title", "track_description", "target_audience", 
                           "prerequisites", "time_commitment", "modules"]
        
        for field in required_fields:
            if field not in parsed:
                print(f"Missing required field in JSON response: {field}")
                # Add the missing field with a placeholder value
                if field == "prerequisites" or field == "modules":
                    parsed[field] = []
                else:
                    parsed[field] = f"No {field.replace('_', ' ')} provided"
        
        # Ensure modules have all required fields
        if "modules" in parsed and parsed["modules"]:
            for i, module in enumerate(parsed["modules"]):
                module_fields = ["title", "description", "learning_objectives", 
                                "key_concepts", "resources", "practice_activities"]
                
                for field in module_fields:
                    if field not in module:
                        print(f"Module {i+1} missing field: {field}")
                        if field in ["learning_objectives", "key_concepts", "resources", "practice_activities"]:
                            module[field] = []
                        else:
                            module[field] = f"No {field.replace('_', ' ')} provided"
                
                # Check resources
                if "resources" in module:
                    for j, resource in enumerate(module["resources"]):
                        resource_fields = ["title", "description", "url", "type", "time_estimate"]
                        for field in resource_fields:
                            if field not in resource:
                                print(f"Resource {j+1} in module {i+1} missing field: {field}")
                                if field == "type":
                                    resource[field] = "article"
                                else:
                                    resource[field] = f"No {field.replace('_', ' ')} provided"
        
        return json.dumps(parsed)
    except Exception as e:
        print(f"Failed to extract JSON: {str(e)}")
        return None

# Function to use classifier model to detect specific exam type
def classify_law_input(user_input):
    try:
        client = get_together_client()
        if not client:
            return "General legal studies"
            
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[
                {"role": "system", "content": "You are a classifier that identifies specific law exam types or legal interest areas."},
                {"role": "user", "content": f"Classify the following text into a specific law exam type or legal interest area:\n\n{user_input}"}
            ],
            temperature=0.3,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Classification error: {str(e)}")
        return "General legal studies"

# Verify link validity function (simplified without making actual HTTP requests)
def is_likely_valid_url(url):
    # Check if URL has a proper format and domain
    if not url.startswith(("http://", "https://")):
        return False
    
    # Check if URL has a domain
    parts = url.split("/")
    if len(parts) < 3:
        return False
    
    domain = parts[2]
    # Check if domain seems reasonable
    if "." not in domain or len(domain) < 4:
        return False
    
    return True

# User token management functions
def award_tokens(user_id, amount):
    """Award tokens to a user for completing activities"""
    try:
        # In a real implementation, this would update a database
        return {"user_id": user_id, "tokens_awarded": amount, "success": True}
    except Exception as e:
        return {"error": str(e), "success": False}

def generate_user_id():
    """Generate a unique user ID based on session and time"""
    return hashlib.md5(f"{datetime.now().isoformat()}_{random.random()}".encode()).hexdigest()[:16]

# Flask routes
@app.route('/api/generate_learning_track', methods=['POST'])
def api_generate_learning_track():
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    category = data.get('category', 'For Fun')
    specific_area = data.get('specific_area', 'General Law')
    
    # Classify the specific area if it's too general
    if len(specific_area.split()) <= 2:
        specific_classification = classify_law_input(specific_area)
        if specific_classification and len(specific_classification.split()) > 2:
            specific_area = specific_classification
    
    # Generate the learning track
    track_data = generate_learning_track(category, specific_area)
    
    if "error" in track_data:
        return jsonify(track_data), 500
    
    # Generate a track ID for future reference
    track_id = hashlib.md5(f"{category}_{specific_area}_{datetime.now().isoformat()}".encode()).hexdigest()[:16]
    track_data["track_id"] = track_id
    
    return jsonify(track_data)

@app.route('/api/get_quiz_questions', methods=['POST'])
def api_get_quiz_questions():
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    module_data = data.get('module')
    if not module_data:
        return jsonify({"error": "No module data provided"}), 400
    
    try:
        # Convert dict to Module
        module_obj = Module(**module_data)
        num_questions = int(data.get('num_questions', 3))
        
        # Generate quiz questions
        questions = generate_quiz_questions(module_obj, num_questions)
        
        # Convert to dict for JSON response
        questions_data = [q.dict() for q in questions]
        
        return jsonify({"questions": questions_data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/award_tokens', methods=['POST'])
def api_award_tokens():
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    user_id = data.get('user_id')
    if not user_id:
        user_id = generate_user_id()
    
    amount = int(data.get('amount', 1))
    
    result = award_tokens(user_id, amount)
    return jsonify(result)

@app.route('/api/track_progress', methods=['POST'])
def api_track_progress():
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    user_id = data.get('user_id')
    if not user_id:
        user_id = generate_user_id()
    
    track_id = data.get('track_id', 'default_track')
    module_title = data.get('module_title')
    objective_index = data.get('objective_index')
    is_completed = data.get('is_completed', True)
    
    # In a real implementation, this would update a database
    response = {
        "user_id": user_id,
        "track_id": track_id,
        "module_title": module_title,
        "objective_index": objective_index,
        "is_completed": is_completed,
        "success": True
    }
    
    # Award tokens if objective was completed
    if is_completed:
        tokens_awarded = 5
        response["tokens_awarded"] = tokens_awarded
    
    return jsonify(response)

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "Law Learning API", "timestamp": datetime.now().isoformat()})

if __name__ == '__main__':
    # For development only - use a production WSGI server in production
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))