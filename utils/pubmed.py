import io
import os
from lxml import etree
from langchain_community.document_loaders.base import BaseLoader
from langchain.schema.document import Document

class PubmedEntry(Document):
    def __init__(self) -> None:
        super().__init__(page_content='', metadata={})

    def add_abstract_text(self, text):
        self.page_content += text
        
    def add_metadata_value(self, key, value):
        self.metadata[key] = str(value)
        
    def extend_metadata_value(self, key, value, delim=', '):
        cv = self.metadata.get(key)
        if cv:
            self.metadata[key] = cv + delim + str(value)
        else:
            self.metadata[key] = str(value)
    
class PubmedXmlEntryBuilder:
    
    class Author:
        def __init__(self):
            self.last_name = None
            self.first_name = 'None'
            self.collective_name = None
            
        def __str__(self):
            if self.last_name:
                return self.last_name + ', ' + self.first_name
            else:
                return self.collective_name
    
    def __init__(self):
        self.currrent_datum = ''
        self.documents = []
        self.current_document = PubmedEntry()
        self.current_author = PubmedXmlEntryBuilder.Author()
        self.abstract = False
        self.stack = []
        
    def start(self, tag, attrib):
        self.currrent_datum = ''
        self.stack.append((tag, dict(attrib)))

    def end(self, tag):
        match tag:
            case 'MedlineCitation':
                self.documents.append(self.current_document)
                self.current_document = PubmedEntry()
                
            case 'AbstractText':
                self.current_document.add_abstract_text(self.currrent_datum)
                self.abstract = False
            
            case 'ForeName':
                self.current_author.first_name = self.currrent_datum
            
            case 'LastName':
                self.current_author.last_name = self.currrent_datum
                
            case 'CollectiveName':
                self.current_author.collective_name = self.currrent_datum
                
            case 'Author':
                try:
                    self.current_document.extend_metadata_value('authors', self.current_author, delim=' and ')
                except Exception as e:
                    print(self.current_document)
                    raise e
                self.current_author = PubmedXmlEntryBuilder.Author()
                                
            case 'PMID':
                if self.stack[-2][0] == 'MedlineCitation':
                    self.current_document.add_metadata_value('pmid', self.currrent_datum)
                
            case 'Title':
                self.current_document.add_metadata_value('journal', self.currrent_datum)
                
            case 'ArticleTitle':
                self.current_document.add_metadata_value('title', self.currrent_datum)
                
            case 'Keyword':
                self.current_document.extend_metadata_value('keywords', self.currrent_datum, delim=', ')
                
            case 'Year':
                self.current_document.add_metadata_value('year', self.currrent_datum)
                
            case 'ELocationID':
                if self.stack[-1][1]['EIdType'] == 'doi':
                    self.current_document.add_metadata_value('doi', self.currrent_datum)
                
            case _:
                pass
            
        self.stack.pop()
            
    def data(self, data):
        self.currrent_datum += data
        
    def comment(self, text):
        pass
        
    def close(self):
        return self.documents
    
class PubmedXmlLoader(BaseLoader):
    def __init__(self, file):
        if type(file) == str:
            if not os.path.isfile(file):
                raise Exception(f'path {file} does not exist')
            self._file = file
            self._stream = None
        elif issubclass(type(file), io.IOBase):
            self._stream = file
            self._file = None
        else:
            raise Exception('file must be a path of stream')
    
    def load(self):
        parser = etree.XMLParser(target=PubmedXmlEntryBuilder(), remove_blank_text=True)
        return etree.parse(self._file, parser)


if __name__ == '__main__':
    loader = PubmedXmlLoader('test/poop.xml')
    docs = loader.load()
    print(len(docs))
