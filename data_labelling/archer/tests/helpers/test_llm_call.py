import pytest
import json
import requests
from unittest.mock import patch, MagicMock

from helpers.llm_call import llm_call

class TestLLMCall:
    """Tests for the llm_call function."""
    
    def test_llm_call_basic_functionality(self):
        """Test basic functionality of llm_call function."""
        # Mock successful API response
        mock_response = {
            "choices": [{"message": {"content": "Generated text"}}]
        }
        
        # Mock the requests.post method
        with patch('requests.post') as mock_post:
            # Configure the mock
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_response
            
            # Call the function
            result = llm_call(
                messages=[{"role": "user", "content": "Hello"}],
                openrouter_api_key="test-key",
                model="gemini-2.0-flash"
            )
            
            # Assert the result
            assert result == mock_response
            
            # Verify request was made with correct data
            mock_post.assert_called_once()
            # Extract the call arguments
            call_args = mock_post.call_args
            url = call_args[1]['url']
            headers = call_args[1]['headers']
            data = json.loads(call_args[1]['data'])
            
            # Verify URL
            assert url == "https://openrouter.ai/api/v1/chat/completions"
            # Verify headers
            assert headers["Authorization"] == "Bearer test-key"
            assert headers["Content-Type"] == "application/json"
            # Verify payload
            assert data["model"] == "gemini-2.0-flash"
            assert data["messages"] == [{"role": "user", "content": "Hello"}]
            assert data["temperature"] == 0.7  # Default value
    
    def test_missing_api_key(self):
        """Test that function raises an error when API key is missing."""
        with pytest.raises(ValueError) as excinfo:
            llm_call(messages=[{"role": "user", "content": "Hello"}])
        
        assert "OpenRouter API key is required" in str(excinfo.value)
    
    def test_api_error_handling(self):
        """Test handling of API errors."""
        # Mock API error response
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 429  # Rate limit error
            mock_post.return_value.text = '{"error": "Rate limit exceeded"}'
            
            # Attempt to call the function
            with pytest.raises(Exception) as excinfo:
                llm_call(
                    messages=[{"role": "user", "content": "Hello"}],
                    openrouter_api_key="test-key"
                )
            
            # Verify error message
            assert "API call failed with status 429" in str(excinfo.value)
    
    def test_optional_parameters(self):
        """Test that optional parameters are correctly included in request."""
        # Mock successful API response
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"choices": [{"message": {"content": "Response"}}]}
            
            # Call function with all optional parameters
            llm_call(
                messages=[{"role": "user", "content": "Hello"}],
                openrouter_api_key="test-key",
                model="gemini-2.0-flash",
                site_url="https://example.com",
                site_name="Test Site",
                temperature=0.5,
                max_tokens=100,
                response_format={"type": "json_object"},
                tools=[{"type": "function", "function": {"name": "test_function"}}]
            )
            
            # Verify request contains all parameters
            call_args = mock_post.call_args
            headers = call_args[1]['headers']
            data = json.loads(call_args[1]['data'])
            
            # Verify headers contain site info
            assert headers["HTTP-Referer"] == "https://example.com"
            assert headers["X-Title"] == "Test Site"
            
            # Verify payload contains all parameters
            assert data["model"] == "gemini-2.0-flash"
            assert data["temperature"] == 0.5
            assert data["max_tokens"] == 100
            assert data["response_format"] == {"type": "json_object"}
            assert data["tools"] == [{"type": "function", "function": {"name": "test_function"}}]
    
    def test_streaming_response(self):
        """Test that streaming parameter returns the response object directly."""
        # Create a mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        
        # Mock the requests.post function
        with patch('requests.post', return_value=mock_response) as mock_post:
            result = llm_call(
                messages=[{"role": "user", "content": "Hello"}],
                openrouter_api_key="test-key",
                stream=True
            )
            
            # For streaming, the function should return the response object
            assert result == mock_response
            # Verify stream=True was in the request
            called_args = mock_post.call_args[1]['data']
            assert json.loads(called_args)["stream"] == True
    
    def test_response_parsing(self):
        """Test parsing of JSON responses."""
        # Mock a complex JSON response
        mock_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gemini-2.0-flash",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello, how can I help you today?"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 10,
                "total_tokens": 20
            }
        }
        
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_response
            
            # Call the function
            result = llm_call(
                messages=[{"role": "user", "content": "Hello"}],
                openrouter_api_key="test-key"
            )
            
            # Assert full response structure is preserved
            assert result["id"] == "chatcmpl-123"
            assert result["object"] == "chat.completion"
            assert result["choices"][0]["message"]["content"] == "Hello, how can I help you today?"
            assert result["usage"]["total_tokens"] == 20
    
    def test_http_timeout(self):
        """Test handling of HTTP timeouts."""
        # Simulate a request timeout
        with patch('requests.post') as mock_post:
            mock_post.side_effect = requests.exceptions.Timeout("Connection timed out")
            
            # Attempt to call the function
            with pytest.raises(requests.exceptions.Timeout) as excinfo:
                llm_call(
                    messages=[{"role": "user", "content": "Hello"}],
                    openrouter_api_key="test-key"
                )
            
            assert "Connection timed out" in str(excinfo.value)
    
    def test_connection_error(self):
        """Test handling of connection errors."""
        # Simulate a connection error
        with patch('requests.post') as mock_post:
            mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")
            
            # Attempt to call the function
            with pytest.raises(requests.exceptions.ConnectionError) as excinfo:
                llm_call(
                    messages=[{"role": "user", "content": "Hello"}],
                    openrouter_api_key="test-key"
                )
            
            assert "Connection refused" in str(excinfo.value)
    
    def test_malformed_response(self):
        """Test handling of malformed API responses."""
        # Mock response with invalid JSON
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            # When json() is called, raise a JSONDecodeError
            mock_post.return_value.json.side_effect = json.JSONDecodeError("Expecting value", "", 0)
            mock_post.return_value.text = "Not a JSON response"
            
            # Attempt to call the function
            with pytest.raises(json.JSONDecodeError):
                llm_call(
                    messages=[{"role": "user", "content": "Hello"}],
                    openrouter_api_key="test-key"
                ) 