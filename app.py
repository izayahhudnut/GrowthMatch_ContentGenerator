from typing import Any, Dict, List, Type
from flask import Flask, request, jsonify
import instructor
from config.setting import get_settings
from openai import OpenAI
from pydantic import BaseModel, Field
import json
import logging
import os




app = Flask(__name__)

class LLMFactory:
    def __init__(self, provider: str):
        self.provider = provider
        self.settings = getattr(get_settings(), provider)
        self.client = self._initialize_client()

    def _initialize_client(self) -> Any:
        client_initializers = {
            "openai": lambda s: instructor.from_openai(OpenAI(api_key=s.api_key))
        }

        initializer = client_initializers.get(self.provider)
        if initializer:
            return initializer(self.settings)
        raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def create_completion(
        self, response_model: Type[BaseModel], messages: List[Dict[str, str]], **kwargs
    ) -> Any:
        completion_params = {
            "model": kwargs.get("model", self.settings.default_model),
            "temperature": kwargs.get("temperature", self.settings.temperature),
            "max_retries": kwargs.get("max_retries", self.settings.max_retries),
            "max_tokens": kwargs.get("max_tokens", self.settings.max_tokens),
            "response_model": response_model,
            "messages": messages,
        }
        return self.client.chat.completions.create(**completion_params)




class SocialMediaPost(BaseModel):
    """Social media post generated from AI transcript and topic"""
    title: str = Field(
        ...,
        max_length=70,
        description=(
            "CONCISE TITLE (70 CHARACTER MAX). MUST USE SENTENCE CASE. "
            "First word capitalized, rest lowercase except proper nouns/acronyms. "
            "BAD EXAMPLE: 'Adapting to Evolving B2B Buyer Behavior' "
            "GOOD EXAMPLE: 'Adapting to evolving B2B buyer behavior'"
        )
    )
    
    body: str = Field(
        ...,
        min_length=500,  # Targets ~100 words (avg 5 chars/word + spaces)
        max_length=2100, # Targets ~350 words
        description=(
            "STRUCTURED CONTENT (100-350 WORDS):\n"
            "1. [HOOK] <150 characters on first line\n"
            "2. [BODY] Concise paragraphs <200 characters each\n"
            "3. [CONCLUSION] Strong ending <200 characters\n"
            "FORMAT:\n"
            "Hook text here\n"
            "\n"
            "Paragraph 1\n"
            "\n"
            "Paragraph 2\n"
            "\n"
            "Final conclusion/question"
        )
    )
    
    hashtags: List[str] = Field(
        ...,
        min_items=3,
        max_items=3,
        description=(
            "EXACTLY 3 HASHTAGS:\n"
            "1. #broad_subject (common)\n"
            "2. #specific_term (from transcript)\n"
            "3. #CustomTerm (unique blend)"
        )
    )
   
class BlogPost(BaseModel):
    """SEO-optimized blog post generated from AI transcript and topic"""
    title: str = Field(
        ...,
        max_length=120,
        description=(
            "CATCHY, SEO-OPTIMIZED TITLE (120 CHARACTER MAX). "
            "Must include primary keyword naturally. "
            "Use title case formatting."
        )
    )
    
    contentBody: str = Field(
        ...,
        min_length=2800,  # Targets ~700 words
        max_length=4000,  # Targets ~1000 words
        description=(
            "STRUCTURED MARKDOWN CONTENT:\n"
            "1. SEO SECTION (frontmatter):\n"
            "   - Focus Keyword: [primary keyword]\n"
            "   - Meta Title: [60 character max]\n"
            "   - Meta Description: [160 character max]\n"
            "   - URL: [slug suggestion]\n"
            "2. CONTENT BODY:\n"
            "   - H1 Headline\n"
            "   - Minimum 3 H2 sections\n"
            "   - Use H3s where appropriate\n"
            "   - Include 1-2 lists\n"
            "   - Conclusion with summary\n"
            "FORMAT:\n"
            "---\n"
            "Focus Keyword: ...\n"
            "Meta Title: ...\n"
            "Meta Description: ...\n"
            "URL: ...\n"
            "---\n\n"
            "# Main Headline\n\n"
            "## Section 1\n\n"
            "Content...\n\n"
            "## Section 2\n\n"
            "- List item\n- List item\n\n"
            "### Subsection\n\n"
            "Content...\n\n"
            "## Conclusion\n\n"
            "Final summary..."
        )
    )
    
    hashtags: List[str] = Field(
        ...,
        min_items=3,
        max_items=5,
        description=(
            "3-5 HASHTAGS FOR SOCIAL SHARING:\n"
            "Mix of industry topics and specific terms"
        )
    )

def PostGenerator(request_data: dict):  
      # Sanitize the keys by removing problematic characters
    sanitized_data = {}
    for key, value in request_data.items():
        new_key = key.replace("'", "").replace('"', "")
        sanitized_data[new_key] = value

    system_content = f'''
        You are an expert senior copywriter and social media marketer who helps business leaders and subject matter experts convert their spoken words into polished, 
        clear and articulate text content. Your core function is to turn AI transcripts from recorded calls into social media marketing posts that establish the writer 
        as a thought leader in their space, educate customers and prospects, and/or market the company the leader or subject matter expert works for.
     
        <Context>
            Please review the various pieces of context below on the creator, their company, their brand, their social post style guide, and more.

            <company_name>
            Company Name: {sanitized_data.get("6.Name", "")}
            </company_name>

            <company_description>
            Company Description: {sanitized_data.get("6.Company Description", "")}
            </company_description>

            <creator_name>
            Creator Name: {sanitized_data.get("11.Full Name", "")}
            </creator_name>

            Please read the bio below of the creator who will be seen as the author of this social post. This creator is also the person who recorded what is in the AI transcript.

            <creator_bio>
            Creator Bio: {sanitized_data.get("11.Bio", "")}
            </creator_bio>

            <optional_context>
            The items below may or may not have values. If they're empty just ignore them when creating the social post text.

            Brand Values: {sanitized_data.get("7.Brand Values", "")}
            Brand Personalities: {sanitized_data.get("7.Brand Personalities", "")}
            Tone of Voice Principles: {sanitized_data.get("7.Tone of Voice Principles", "")}
            Language Dos: {sanitized_data.get("7.Language Dos", "")}
            Language Donts: {sanitized_data.get("7.Language Donts", "")}
            Audience Adapatations: {sanitized_data.get("7.Audience Adaptations", "")}
            On Brand Examples: {sanitized_data.get("7.On Brand Examples", "")}
            Off Brand Examples: {sanitized_data.get("7.Off Brand (Bad) Examples", "")}
            Narrative Style: {sanitized_data.get("7.Narrative and Storytelling Style", "")}
            Key Messages: {sanitized_data.get("7.Key Messages", "")}
            Social Post Style Guide: {sanitized_data.get("7.Social Post Writing Style Guide", "")}
            Audience Information: {sanitized_data.get("14.array", "")}
            Solutions Information: {sanitized_data.get("13.array", "")}
            </optional_context>
            </context>
            
            <examples>
            <example1>
            {sanitized_data.get("11.Social Post Sample 1", "")}
            </example1>
            <example2>
            {sanitized_data.get("11.Social Post Sample 2", "")}
            </example2>
            <example3>
            {sanitized_data.get("11.Social Post Sample 3", "")}
            </example3>
            </examples>
    '''

    user_content = f'''
        Please write a social media post using your system instructions and given the AI transcript and topic below:

        <ai_transcript>
        AI Transcript: {sanitized_data.get("2.AI Transcript Rough", "")}
        </ai_transcript>
        <topic>
        Topic Name: {sanitized_data.get("12.Topic Name", "")}
        </topic>
    '''

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

    # Generate the List
    llm = LLMFactory("openai")
    completion = llm.create_completion(response_model=SocialMediaPost, messages=messages)

    return {
        "title": completion.title,
        "body": completion.body,
        "hashtags": completion.hashtags,
    }

def BlogPostGenerator(request_data: dict):
    # Sanitize the keys
    sanitized_data = {}
    for key, value in request_data.items():
        new_key = key.replace("'", "").replace('"', "").replace("`", "")
        sanitized_data[new_key] = value

    system_content = f'''
        You are an expert content strategist and SEO specialist who helps convert spoken-word transcripts into 
        professional, optimized blog posts. Your goal is to create long-form content that ranks well in search engines
        while maintaining natural readability and thought leadership.

        <Context>
            <company_name>
            Company Name: {sanitized_data.get("6.Name", "")}
            </company_name>

            <company_description>
            Company Description: {sanitized_data.get("6.Company Description", "")}
            </company_description>

            <creator_name>
            Creator Name: {sanitized_data.get("11.Full Name", "")}
            </creator_name>

            <creator_bio>
            Creator Bio: {sanitized_data.get("11.Bio", "")}
            </creator_bio>

            <style_guide>
            Long Form Style Guide: {sanitized_data.get("7.Long Form Writing Style Guide", "")}
            </style_guide>

            <optional_context>
            Brand Values: {sanitized_data.get("7.Brand Values", "")}
            Tone Guidelines: {sanitized_data.get("7.Tone of Voice Principles", "")}
            Audience: {sanitized_data.get("14.array", "")}
            Solutions: {sanitized_data.get("13.array", "")}
            </optional_context>
        </context>

        <examples>
        <example1>
        {sanitized_data.get("11.Long Form Text Sample 1", "")}
        </example1>
        <example2>
        {sanitized_data.get("11.Long Form Text Sample 2", "")}
        </example2>
        <example3>
        {sanitized_data.get("11.Long Form Text Sample 3", "")}
        </example3>
        </examples>
    '''

    user_content = f'''
        Create an SEO-optimized blog post from this transcript and topic:

        <ai_transcript>
        {sanitized_data.get("2.AI Transcript Rough", "")}
        </ai_transcript>

        <topic>
        {sanitized_data.get("12.Topic Name", "")}
        </topic>

        Follow these structural requirements:
        - Start with SEO frontmatter section
        - Use proper heading hierarchy (H1 > H2 > H3)
        - Include at least one bulleted list
        - Keep paragraphs under 5 lines
        - Use transitional phrases between sections
        - Include natural keyword placement
        - Add 3-5 relevant hashtags at end
    '''

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

    llm = LLMFactory("openai")
    completion = llm.create_completion(response_model=BlogPost, messages=messages)

    return {
        "title": completion.title,
        "contentBody": completion.contentBody,
        "hashtags": completion.hashtags,
    }
    
    
@app.route('/generate_blog', methods=['POST'])
def generate_blog():
    try:
        # Input validation
        required_fields = ["2.AI Transcript Rough", "12.Topic Name"]
        for field in required_fields:
            if field not in request.json:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        result = BlogPostGenerator(request.json)
        return jsonify(result)
    except Exception as e:
        logging.error(f"Error generating blog post: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/generate_post', methods=['POST'])
def generate_post():
    try:
        # Print the request data
        print("Request Headers:")
        print(json.dumps(dict(request.headers), indent=2))
        
        print("\nRequest Body:")
        print(json.dumps(request.json, indent=2))

        # Call the PostGenerator function with the request data
        result = PostGenerator(request.json)
        return jsonify(result)
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    logging.info(f'Starting Flask app on port {port}')
    app.run(debug=True, host='0.0.0.0', port=port)