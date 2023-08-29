# Carl v0.0.1 prototype

import urllib.parse
from playwright.async_api import async_playwright
from transformers import BartForConditionalGeneration, BartTokenizer

# The function to fetch search results using Playwright (asynchronous version)
async def fetch_search_results(query, max_results=20):
    content = ""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        search_url = f"https://www.bing.com/search?q={urllib.parse.quote(query)}&PC=U316&FORM=CHROMN"
        
        timeout_value = 600000
        await page.goto(search_url, timeout=timeout_value)
        await page.wait_for_selector(".b_algo")

        for i in range(max_results):
            link_element = await page.query_selector(f".b_algo:nth-child({i + 1}) h2 a")
            if not link_element:
                continue

            link = await link_element.get_attribute("href")
            link_timeout_value = 300000
            await page.goto(link, timeout=link_timeout_value)
            
            main_content_element = await page.query_selector("main") or await page.query_selector("body")
            content = await main_content_element.inner_text() if main_content_element else ""

            if content:
                break

        await browser.close()
    return content if content else "No suitable content found. The application will not proceed."

# The Carl class for preprocessing and summarizing content
class Carl:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)

    def preprocess_content(self, content):
        inputs = self.tokenizer([content], max_length=1024, return_tensors="pt", truncation=True)
        return inputs

    def generate_summary(self, content):
        inputs = self.preprocess_content(content)
        summary_ids = self.model.generate(inputs["input_ids"], num_beams=4, max_length=250, early_stopping=True)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    async def fetch_and_summarize(self, query):
        content = await fetch_search_results(query)
        summary = self.generate_summary(content)
        return summary

# Example usage
carl = Carl()
summary = await carl.fetch_and_summarize("Quantum Mechanics")
print(summary)
