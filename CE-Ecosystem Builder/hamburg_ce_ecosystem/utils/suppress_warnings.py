"""Utility to suppress verbose output from third-party libraries."""
import logging
import warnings
import os
import sys


class FilteredOutput:
    """Filter stdout/stderr to suppress specific messages while preserving tqdm compatibility."""
    
    def __init__(self, original_stream):
        self.original_stream = original_stream
        self.suppress_patterns = [
            'Found providers',
            'If it was not intended please specify the model provider',
            'Attempt 1 failed:',
            'Attempt 2 failed:',
            'Attempt 3 failed:',
            'Page.goto:',
            'Call log:',
            'navigating to',
            'net::ERR_',
            'Error during chain execution:',
            'Invalid json output:',
            'For troubleshooting, visit:',
            'OUTPUT_PARSING_FAILURE',
            '<think>',
            '</think>',
            '✨ Try enhanced version of ScrapegraphAI',
            'Es sieht so aus',
            'Das ist eine',
            'This is a massive',
            'This appears to be',
            'Hier sind einige',
            'I\'ll try to summarize',
            'Overall, this text',
        ]
    
    def write(self, text):
        # Aggressive filtering of error blocks
        # Check if entire text chunk should be suppressed
        if any(pattern in text for pattern in self.suppress_patterns):
            # This is an error/noise block, suppress it entirely
            return len(text)
        
        # If not suppressed, write it
        return self.original_stream.write(text)
    
    def flush(self):
        return self.original_stream.flush()
    
    def isatty(self):
        return self.original_stream.isatty()
    
    def fileno(self):
        return self.original_stream.fileno()
    
    def __getattr__(self, name):
        # Forward any other attributes to original stream
        return getattr(self.original_stream, name)


def suppress_verbose_output():
    """Suppress verbose logging and print output from scrapegraphai and playwright."""
    # Suppress warnings
    warnings.filterwarnings('ignore')
    
    # Point Playwright to the correct browser location
    # Remove the incorrect PLAYWRIGHT_BROWSERS_PATH setting
    if 'PLAYWRIGHT_BROWSERS_PATH' in os.environ:
        del os.environ['PLAYWRIGHT_BROWSERS_PATH']
    
    # Suppress verbose logging from various libraries
    logging.getLogger('scrapegraphai').setLevel(logging.CRITICAL)
    logging.getLogger('playwright').setLevel(logging.CRITICAL)
    logging.getLogger('httpx').setLevel(logging.CRITICAL)
    logging.getLogger('httpcore').setLevel(logging.CRITICAL)
    
    # Filter stdout and stderr to suppress print statements and error messages
    if not isinstance(sys.stdout, FilteredOutput):
        sys.stdout = FilteredOutput(sys.stdout)
    if not isinstance(sys.stderr, FilteredOutput):
        sys.stderr = FilteredOutput(sys.stderr)
