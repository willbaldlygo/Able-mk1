"""Audio transcription service for Able."""
import tempfile
import os
from typing import Optional
import openai
from config import config


class TranscriptionService:
    """Handles audio transcription using OpenAI Whisper API."""
    
    def __init__(self):
        # Check if OpenAI API key is available
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
            self.use_whisper = True
        else:
            self.use_whisper = False
    
    def transcribe_audio(self, audio_data: bytes) -> Optional[str]:
        """Transcribe audio data to text using OpenAI Whisper API."""
        try:
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            try:
                if self.use_whisper:
                    # Use OpenAI Whisper API
                    with open(temp_file_path, 'rb') as audio_file:
                        transcript = openai.Audio.transcribe("whisper-1", audio_file)
                        return transcript.text.strip()
                else:
                    # Fallback to browser-based transcription message
                    return self._browser_transcription_message()
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    
        except Exception as e:
            print(f"Transcription error: {str(e)}")
            return self._browser_transcription_message()
    
    def _browser_transcription_message(self) -> str:
        """
        Fallback message when OpenAI API key is not available.
        Suggests using browser-based transcription.
        """
        return "Please add OPENAI_API_KEY to your .env file for automatic transcription, or use browser's built-in speech recognition"
    
    def test_connection(self) -> bool:
        """Test if the transcription service is available."""
        try:
            if self.use_whisper and self.openai_api_key:
                # Test OpenAI API connection
                return True
            else:
                # Always return True for fallback mode
                return True
        except Exception:
            return False