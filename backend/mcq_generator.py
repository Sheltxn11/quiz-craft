"""
MCQ Generation Pipeline
Implements Steps 3-5: Sampling/Compression, Prompt Construction, and LLM Integration
"""

import json
import logging
import asyncio
from typing import List, Dict, Any, Optional
import tiktoken
import google.generativeai as genai
from embedding_generator import EmbeddingGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCQGenerator:
    """
    Handles MCQ generation from text chunks using LLM integration.
    """
    
    def __init__(self, 
                 gemini_api_key: str = None,
                 token_limit: int = 3000,
                 max_questions_per_batch: int = 5):
        """
        Initialize the MCQ generator.
        
        Args:
            gemini_api_key: Google Gemini API key
            token_limit: Maximum tokens for LLM context (default: 3000 for Gemini)
            max_questions_per_batch: Maximum questions per API call
        """
        self.token_limit = token_limit
        self.max_questions_per_batch = max_questions_per_batch
        
        # Initialize tokenizer (using GPT-4 tokenizer as proxy for Gemini)
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        
        # Initialize Gemini
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            # Use the correct model name for Gemini 1.5
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            logger.info("✅ Gemini API configured with gemini-1.5-flash")
        else:
            self.model = None
            logger.warning("⚠️ No Gemini API key provided")
        
        # Initialize embedding generator for chunk retrieval
        self.embedding_generator = EmbeddingGenerator()
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logger.warning(f"Token counting failed, using word count estimate: {e}")
            # Fallback: rough estimate (1 token ≈ 0.75 words)
            return int(len(text.split()) * 1.33)
    
    def compress_chunk(self, chunk_content: str, target_tokens: int = 50) -> str:
        """
        Compress a chunk to target token count using Gemini.
        
        Args:
            chunk_content: Original chunk content
            target_tokens: Target token count for compressed version
            
        Returns:
            Compressed chunk content
        """
        if not self.model:
            # Fallback: simple truncation
            words = chunk_content.split()
            target_words = int(target_tokens * 0.75)  # Rough conversion
            return ' '.join(words[:target_words]) + "..." if len(words) > target_words else chunk_content
        
        try:
            prompt = f"Summarize this text in ≤ {target_tokens} tokens, preserving key information:\n\n{chunk_content}"
            response = self.model.generate_content(prompt)
            compressed = response.text.strip()
            
            # Verify compression worked
            if self.count_tokens(compressed) <= target_tokens * 1.2:  # 20% tolerance
                return compressed
            else:
                # Fallback to truncation if compression failed
                words = chunk_content.split()
                target_words = int(target_tokens * 0.75)
                return ' '.join(words[:target_words]) + "..."
                
        except Exception as e:
            logger.warning(f"Chunk compression failed: {e}")
            # Fallback: simple truncation
            words = chunk_content.split()
            target_words = int(target_tokens * 0.75)
            return ' '.join(words[:target_words]) + "..." if len(words) > target_words else chunk_content
    
    def sample_and_compress_chunks(self, 
                                  candidate_chunks: List[Dict[str, Any]],
                                  token_budget: int = None) -> str:
        """
        Step 3: Compress or sample chunks to fit token limit.
        
        Args:
            candidate_chunks: List of chunk dictionaries from Qdrant
            token_budget: Available token budget (defaults to self.token_limit)
            
        Returns:
            Compressed context string
        """
        if not candidate_chunks:
            logger.warning("No candidate chunks provided")
            return ""
        
        token_budget = token_budget or self.token_limit
        logger.info(f"Processing {len(candidate_chunks)} chunks with {token_budget} token budget")
        
        # Step 3.1: Estimate total tokens
        total_tokens = 0
        chunk_tokens = []
        
        for chunk in candidate_chunks:
            content = chunk.get('content', '')
            tokens = self.count_tokens(content)
            chunk_tokens.append(tokens)
            total_tokens += tokens
        
        logger.info(f"Total tokens in chunks: {total_tokens}")
        
        # Step 3.2: Check if compression/sampling is needed
        if total_tokens <= token_budget:
            logger.info("Chunks already fit within token budget")
            return "\n\n".join([chunk.get('content', '') for chunk in candidate_chunks])
        
        # Step 3.3: Compression strategy
        logger.info("Applying compression and sampling...")
        
        # Target tokens per chunk after compression
        target_tokens_per_chunk = min(50, token_budget // len(candidate_chunks))
        
        compressed_chunks = []
        used_tokens = 0
        
        for i, chunk in enumerate(candidate_chunks):
            if used_tokens >= token_budget:
                break
                
            content = chunk.get('content', '')
            
            # Compress chunk if needed
            if chunk_tokens[i] > target_tokens_per_chunk:
                compressed_content = self.compress_chunk(content, target_tokens_per_chunk)
            else:
                compressed_content = content
            
            # Check if we can fit this chunk
            chunk_token_count = self.count_tokens(compressed_content)
            if used_tokens + chunk_token_count <= token_budget:
                compressed_chunks.append(compressed_content)
                used_tokens += chunk_token_count
            else:
                # Try to fit a smaller version
                remaining_budget = token_budget - used_tokens
                if remaining_budget > 20:  # Minimum viable chunk size
                    final_compressed = self.compress_chunk(content, remaining_budget - 5)
                    compressed_chunks.append(final_compressed)
                break
        
        # Step 3.4: Uniform sampling if still too many chunks
        if len(compressed_chunks) > token_budget // 30:  # Rough estimate: 30 tokens per chunk minimum
            step = len(compressed_chunks) // (token_budget // 30)
            compressed_chunks = compressed_chunks[::max(1, step)][:token_budget // 30]
        
        context_text = "\n\n".join(compressed_chunks)
        final_tokens = self.count_tokens(context_text)
        
        logger.info(f"Final context: {len(compressed_chunks)} chunks, {final_tokens} tokens")
        return context_text
    
    def construct_prompt(self, 
                        context_text: str,
                        num_questions: int,
                        difficulty: str = "medium",
                        bloom_level: str = "application") -> tuple:
        """
        Step 4: Build dynamic prompt for LLM.
        
        Args:
            context_text: Compressed context from chunks
            num_questions: Number of questions to generate
            difficulty: Question difficulty level
            bloom_level: Bloom's taxonomy level
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        if not context_text.strip():
            raise ValueError("Context text is empty - cannot construct prompt")
        
        # System prompt
        system_prompt = """You are an expert exam-setter for competitive exams. You create high-quality multiple-choice questions that test deep understanding and critical thinking."""
        
        # User prompt with dynamic content
        user_prompt = f"""Generate {num_questions} multiple-choice questions based on the text below.

Content:
\"\"\"
{context_text}
\"\"\"

Instructions:
- Each question should have 4 options (A–D)
- Mark the correct answer clearly
- Add a 1-line explanation for the correct answer
- Difficulty: {difficulty}
- Bloom's Taxonomy Level: {bloom_level}
- Questions should test understanding, not just memorization
- Ensure questions are based strictly on the provided content

Return a valid JSON array in this exact format:
[
  {{
    "question": "What is the main concept discussed in the text?",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "answer": "B",
    "explanation": "The text clearly states that Option B is correct because..."
  }}
]

Important: Return ONLY the JSON array, no additional text or formatting."""
        
        return system_prompt, user_prompt
    
    async def call_gemini_async(self, system_prompt: str, user_prompt: str) -> str:
        """
        Call Gemini API asynchronously.
        
        Args:
            system_prompt: System instruction
            user_prompt: User query
            
        Returns:
            Generated response text
        """
        if not self.model:
            raise ValueError("Gemini API not configured")
        
        try:
            # Combine prompts for Gemini
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            # Generate content
            response = self.model.generate_content(full_prompt)
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            raise e
    
    def parse_mcq_response(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Parse LLM response into structured MCQ format.
        
        Args:
            response_text: Raw response from LLM
            
        Returns:
            List of MCQ dictionaries
        """
        try:
            # Clean response text
            cleaned_response = response_text.strip()
            
            # Remove any markdown formatting
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            
            cleaned_response = cleaned_response.strip()
            
            # Parse JSON
            mcqs = json.loads(cleaned_response)
            
            # Validate structure
            if not isinstance(mcqs, list):
                raise ValueError("Response is not a JSON array")
            
            for i, mcq in enumerate(mcqs):
                required_fields = ['question', 'options', 'answer', 'explanation']
                for field in required_fields:
                    if field not in mcq:
                        raise ValueError(f"MCQ {i+1} missing required field: {field}")
                
                if not isinstance(mcq['options'], list) or len(mcq['options']) != 4:
                    raise ValueError(f"MCQ {i+1} must have exactly 4 options")
                
                if mcq['answer'] not in ['A', 'B', 'C', 'D']:
                    raise ValueError(f"MCQ {i+1} answer must be A, B, C, or D")
            
            logger.info(f"Successfully parsed {len(mcqs)} MCQs")
            return mcqs
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            logger.error(f"Response text: {response_text[:500]}...")
            raise ValueError(f"Invalid JSON response from LLM: {e}")
        except Exception as e:
            logger.error(f"MCQ parsing failed: {e}")
            raise e

    async def generate_mcqs_batch(self,
                                 context_text: str,
                                 num_questions: int,
                                 difficulty: str = "medium",
                                 bloom_level: str = "application") -> List[Dict[str, Any]]:
        """
        Step 5: Generate MCQs by calling LLM in batches.

        Args:
            context_text: Compressed context text
            num_questions: Total number of questions to generate
            difficulty: Question difficulty level
            bloom_level: Bloom's taxonomy level

        Returns:
            List of generated MCQs
        """
        if not context_text.strip():
            raise ValueError("Context text is empty")

        all_mcqs = []

        # Process in batches to avoid token limits
        for i in range(0, num_questions, self.max_questions_per_batch):
            batch_size = min(self.max_questions_per_batch, num_questions - i)

            logger.info(f"Generating batch {i//self.max_questions_per_batch + 1}: {batch_size} questions")

            # Construct prompt for this batch
            system_prompt, user_prompt = self.construct_prompt(
                context_text=context_text,
                num_questions=batch_size,
                difficulty=difficulty,
                bloom_level=bloom_level
            )

            # Call LLM
            try:
                response = await self.call_gemini_async(system_prompt, user_prompt)
                batch_mcqs = self.parse_mcq_response(response)
                all_mcqs.extend(batch_mcqs)

                logger.info(f"Successfully generated {len(batch_mcqs)} MCQs in batch")

            except Exception as e:
                logger.error(f"Batch {i//self.max_questions_per_batch + 1} failed: {e}")
                # Continue with other batches
                continue

        logger.info(f"Total MCQs generated: {len(all_mcqs)}")
        return all_mcqs

    async def generate_mcqs_from_search(self,
                                       query: str,
                                       num_questions: int = 5,
                                       difficulty: str = "medium",
                                       bloom_level: str = "application",
                                       search_limit: int = 10,
                                       score_threshold: float = 0.3,
                                       pdf_filename: str = None,
                                       page_range: tuple = None) -> Dict[str, Any]:
        """
        Complete pipeline: Search chunks → Compress → Generate MCQs.

        Args:
            query: Search query for relevant chunks
            num_questions: Number of MCQs to generate
            difficulty: Question difficulty
            bloom_level: Bloom's taxonomy level
            search_limit: Maximum chunks to retrieve
            score_threshold: Minimum similarity score
            pdf_filename: Filter by specific PDF
            page_range: Filter by page range (min_page, max_page)

        Returns:
            Dictionary with MCQs and metadata
        """
        logger.info(f"Starting MCQ generation pipeline for query: '{query}'")

        try:
            # Step 1: Search for relevant chunks
            filter_conditions = None
            if pdf_filename or page_range:
                must_conditions = []

                if pdf_filename:
                    must_conditions.append({"key": "pdf_filename", "match": {"value": pdf_filename}})

                if page_range:
                    min_page, max_page = page_range
                    must_conditions.append({"key": "page", "range": {"gte": min_page, "lte": max_page}})

                filter_conditions = {"must": must_conditions}

            # Search chunks
            if filter_conditions:
                candidate_chunks = self.embedding_generator.search_with_filter(
                    query_text=query,
                    filter_conditions=filter_conditions,
                    limit=search_limit,
                    score_threshold=score_threshold
                )
            else:
                candidate_chunks = self.embedding_generator.search_similar_chunks(
                    query_text=query,
                    limit=search_limit,
                    score_threshold=score_threshold
                )

            if not candidate_chunks:
                return {
                    'error': 'No relevant chunks found for the query',
                    'query': query,
                    'mcqs': [],
                    'metadata': {}
                }

            logger.info(f"Found {len(candidate_chunks)} relevant chunks")

            # Step 2: Sample and compress chunks
            context_text = self.sample_and_compress_chunks(candidate_chunks)

            if not context_text.strip():
                return {
                    'error': 'No usable content after compression',
                    'query': query,
                    'mcqs': [],
                    'metadata': {}
                }

            # Step 3: Generate MCQs
            mcqs = await self.generate_mcqs_batch(
                context_text=context_text,
                num_questions=num_questions,
                difficulty=difficulty,
                bloom_level=bloom_level
            )

            # Prepare metadata
            metadata = {
                'query': query,
                'chunks_found': len(candidate_chunks),
                'context_tokens': self.count_tokens(context_text),
                'difficulty': difficulty,
                'bloom_level': bloom_level,
                'pdf_filename': pdf_filename,
                'page_range': page_range,
                'search_filters': filter_conditions is not None,
                'generation_timestamp': logger.info.__self__.name if hasattr(logger.info, '__self__') else 'unknown'
            }

            return {
                'mcqs': mcqs,
                'metadata': metadata,
                'context_preview': context_text[:500] + "..." if len(context_text) > 500 else context_text,
                'chunks_used': [
                    {
                        'chunk_id': chunk['chunk_id'],
                        'page': chunk['page'],
                        'score': chunk.get('score', 0),
                        'content_preview': chunk['content'][:100] + "..."
                    }
                    for chunk in candidate_chunks
                ]
            }

        except Exception as e:
            logger.error(f"MCQ generation pipeline failed: {e}")
            return {
                'error': str(e),
                'query': query,
                'mcqs': [],
                'metadata': {}
            }

def main():
    """Test the MCQ generator"""
    import os
    from dotenv import load_dotenv

    load_dotenv()

    # Initialize generator
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    if not gemini_api_key:
        print("❌ GEMINI_API_KEY not found in environment")
        return

    generator = MCQGenerator(gemini_api_key=gemini_api_key)

    async def test_generation():
        # Test MCQ generation
        result = await generator.generate_mcqs_from_search(
            query="What is the volume of a cube?",
            num_questions=3,
            difficulty="medium",
            bloom_level="application"
        )

        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"✅ Generated {len(result['mcqs'])} MCQs")
            print(f"📊 Metadata: {result['metadata']}")

            for i, mcq in enumerate(result['mcqs'], 1):
                print(f"\n{i}. {mcq['question']}")
                for j, option in enumerate(mcq['options']):
                    print(f"   {chr(65+j)}. {option}")
                print(f"   Answer: {mcq['answer']}")
                print(f"   Explanation: {mcq['explanation']}")

    # Run test
    asyncio.run(test_generation())

if __name__ == "__main__":
    main()
