
import pysolr

solr_address = 'http://185.211.59.100:9997/solr/tax_similar_docs'
solr_userName = 'solr'
solr_password = 'Solr@123'

def solr_commit(documents):
    solr = pysolr.Solr(solr_address, auth=(solr_userName, solr_password), timeout=15)
    solr.add(documents, commit=False, softCommit=True, )
    print('Solr {} documents Commited'.format(len(documents)))


def solr_getRelatedDocs(clause_num):
    solr = pysolr.Solr(solr_address, auth=(solr_userName, solr_password), timeout=15)
    query = "clause_num:" + clause_num
    fields = 'id,text,text_normalized,num,type,type_id'
    response = solr.search(query, rows=1000, fl=fields)
    return response.docs