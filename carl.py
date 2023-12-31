# Carl v0.0.2 
# This version uses dynamic graph knowledge and is much more accurate,
# try questions from this list: https://openstax.org/books/biology-2e/pages/1-critical-thinking-questions
# the model still doesn't have cache or feedback loop


import torch
import urllib.parse
from playwright.async_api import async_playwright
from transformers import BartForConditionalGeneration, BartTokenizer
import spacy

# Function to calculate perplexity for a given content
def calculate_perplexity(model, tokenizer, text):
    inputs = tokenizer([text], return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        log_likelihood = outputs.loss
    perplexity = torch.exp(log_likelihood).item()
    return perplexity

# The function to fetch search results using Playwright (asynchronous version)
async def fetch_search_results(query, max_results=20):
    content = ""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        search_url = f"https://www.bing.com/search?q={urllib.parse.quote(query)}&PC=U316&FORM=CHROMN"
        
        timeout_value = 30000
        await page.goto(search_url, timeout=timeout_value)
        await page.wait_for_selector(".b_algo")

        for i in range(max_results):
            link_element = await page.query_selector(f".b_algo:nth-child({i + 1}) h2 a")
            if not link_element:
                continue

            link = await link_element.get_attribute("href")
            link_timeout_value = 30000
            await page.goto(link, timeout=link_timeout_value)
            
            main_content_element = await page.query_selector("main") or await page.query_selector("body")
            content = await main_content_element.inner_text() if main_content_element else ""

            if content:
                break

        await browser.close()
    return content if content else "No suitable content found. The application will not proceed."

# Dynamic Knowledge Graph class
class DynamicKnowledgeGraph:
    def __init__(self):
        self.graph = {}
        self.nlp = spacy.load("en_core_web_sm")

    def update_graph(self, content):
        doc = self.nlp(content)
        for ent in doc.ents:
            self.graph[ent.text] = ent.label_

    def query(self, entity):
        return self.graph.get(entity, "Entity not found in the knowledge graph.")

# The Carl class for preprocessing and summarizing content
class Carl:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.knowledge_graph = DynamicKnowledgeGraph()

    def preprocess_content(self, content):
        inputs = self.tokenizer([content], max_length=1024, return_tensors="pt", truncation=True)
        return inputs

    def generate_summary(self, content):
        inputs = self.preprocess_content(content)
        summary_ids = self.model.generate(inputs["input_ids"], num_beams=8, max_length=1024, early_stopping=True)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    async def fetch_and_summarize(self, query, max_iterations=3):
        iteration = 0
        while iteration < max_iterations:
            content = await fetch_search_results(query)
            self.knowledge_graph.update_graph(content)
            
            summary = self.generate_summary(content)
            perplexity = calculate_perplexity(self.model, self.tokenizer, summary)

            if perplexity < 20:
                doc = self.knowledge_graph.nlp(query)
                for ent in doc.ents:
                    info = self.get_info_from_graph(ent.text)
                    print(f"Knowledge Graph Info for {ent.text}: {info}")

                return summary, perplexity
            
            iteration += 1
            query += " more information"

        return "Could not find satisfactory content after multiple attempts.", None

    def get_info_from_graph(self, entity):
        return self.knowledge_graph.query(entity)

# Example usage
carl = Carl()
summary, perplexity = await carl.fetch_and_summarize("Name two topics that are likely to be studied by biologists, and two areas of scientific study that would fall outside the realm of biology.")
print(f"Summary:\n{summary}\n\nPerplexity: {perplexity}")
