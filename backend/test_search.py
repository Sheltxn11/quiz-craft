"""
Test script for embedding search functionality
"""

from embedding_generator import EmbeddingGenerator
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def test_search():
    """Test the search functionality"""
    try:
        print("🔍 Testing embedding search functionality...")
        
        # Initialize generator
        generator = EmbeddingGenerator()
        
        # Test queries
        test_queries = [
            "What is the volume of a cube?",
            "How to calculate density?",
            "What is astronomical unit?",
            "Measurement of liquids"
        ]
        
        for query in test_queries:
            print(f"\n📝 Query: '{query}'")
            results = generator.search_similar_chunks(query, limit=3, score_threshold=0.2)
            
            if results:
                print(f"✅ Found {len(results)} results:")
                for i, result in enumerate(results, 1):
                    print(f"  {i}. Score: {result['score']:.3f} | Page: {result['page']} | Tokens: {result['token_count']}")
                    print(f"     Content: {result['content'][:150]}...")
                    print()
            else:
                print("❌ No results found")
        
        print("✅ Search test completed!")
        
    except Exception as e:
        print(f"❌ Error during search test: {e}")

if __name__ == "__main__":
    test_search()
