from crewai_tools import ScrapeWebsiteTool
import os

class Scraper:
    """
    A class that handles the scraping of website content and saving it into a file.

    This class uses the Crew AI tools to extract textual data from a given website URL
    and saves the content into a text file for further processing or use in AI-powered applications.
    """

    def __init__(self):
        """
        Initializes the Scraper class.
        
        This method sets up any necessary parameters or configurations for the scraper.
        """
        pass

    def scrape(self, url: str, filename: str = 'ai.txt', directory: str = '', overwrite: bool = True):
        """
        Scrapes content from a given website URL and saves the extracted text to a file.

        Parameters:
            url (str): The URL of the website to scrape.
            filename (str): The name of the file to save the scraped content. Default is 'ai.txt'.
            directory (str): The directory where the file should be saved. Default is the current directory.
            overwrite (bool): Whether to overwrite the file if it already exists. Default is True.

        Returns:
            None

        Raises:
            Exception: If there is an issue scraping the website or writing the content to a file.
        """
        try:
            # Initialize the ScrapeWebsiteTool with the given URL
            tool = ScrapeWebsiteTool(website_url=url)
            
            # Extract text from the website
            print("Extracting text from the website...")
            text = tool.run()

            # Debug: Print a snippet of the extracted text
            print(f"Extracted text length: {len(text)}")
            if not text.strip():
                print("Warning: No text was extracted.")
                return


            # Set the file path
            abs_directory = os.path.abspath(directory) if directory else os.getcwd()
            abs_file_path = os.path.join(abs_directory, filename)

            # Check if the file exists and handle overwrite logic
            if os.path.exists(abs_file_path) and not overwrite:
                print(f"File already exists and overwrite is disabled: {abs_file_path}")
                return

            # Write the scraped content to the file
            with open(abs_file_path, 'w', encoding='utf-8') as file:
                file.write(text)

            print(f"Content successfully saved to {abs_file_path}")

        except Exception as e:
            # Log and raise an exception if there is an error during scraping or file writing
            print(f"An error occurred: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    # Instantiate the Scraper class
    scraper = Scraper()
    
    # Scrape content from the desired website and save it to 'ai.txt'
    scraper.scrape(url="https://codemate.ai/", filename="ai.txt", overwrite=True)
