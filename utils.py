from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.document_transformers import EmbeddingsRedundantFilter
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

def meta2docs(spss_meta, excluded = ['CNTRYID']):
    docs = []
    for col in  spss_meta.column_names:
        #if col in spss_meta.variable_value_labels:
        if col not in excluded:
            #print(col)
            docs.append(
                Document(
                    page_content=spss_meta.column_names_to_labels[col],
                    metadata={"year": 2022, "original_col_name": col},
                ),
            )
    return docs

def generate_column_name_hints(llm, question):
    prompt1 = f"Please list the typical database column fields, that required to answer the following question: {question}"
    relevant_col_list_msg = llm.invoke(prompt1)
    relevant_col_list = relevant_col_list_msg.content
    return relevant_col_list


def match_column_names(hints_text, retriever):
    filter = EmbeddingsRedundantFilter(embeddings=OpenAIEmbeddings(), similarity_threshold=0.99)
    res = []
    hints = hints_text.split('\n')
    for hint in hints:
        rel_col_docs = retriever.invoke(hint)    
        print([i.page_content for i in rel_col_docs])
        rel_col_docs = filter.transform_documents(rel_col_docs)
        print([i.page_content for i in rel_col_docs])
        res = res+rel_col_docs
    return res

def docs2explanation(docs, meta):
    t = ''
    for idx, doc in enumerate(docs):
        col_name = doc.metadata['original_col_name']
        if col_name in meta.variable_measure.keys() and meta.variable_measure[col_name] != 'unknown':
            scale = meta.variable_measure[col_name]            
            if scale == 'scale':
                scale = 'interval'                
            measure = ' A ' + meta.readstat_variable_types[col_name] + ' variable with ' +scale  + ' scale measure.'
        else:
            measure = ''
        t =  t + str(idx+1)+ '. ' + col_name + " : " + doc.page_content +'.' + measure +'\n'
    t = t + ''
    return t

def gen_code(llm, question, rel_col_docs, meta):
    data_explanation = docs2explanation(rel_col_docs, meta)
    columns = [i.metadata['original_col_name'] for i in rel_col_docs]
    prompt2 = f"Given a dataframe with the following columns {columns}, column meaning: {data_explanation}, can you generate a python code, without sample data, which can answer the following question? the code must contain only one function called 'run', that returns an exact number of type 'float'. Do not write explanation, just code. \nQuestion: {question}"
    res = llm.invoke(prompt2)
    print(res)
    code = res.content.replace('```python','').replace('```','')
    return code

def exec_code(code, df):        
    df2 = df.dropna()
    loc = locals()
    exec(code + "\nr = run(df2)\n", globals(), loc)
    return loc['r']

def pipeline(llm, question, df, meta, col_retriever):
    col_hints = generate_column_name_hints(llm, question)
    print(col_hints)
    rel_col_docs = match_column_names(col_hints, col_retriever)    
    print([i.page_content for i in rel_col_docs])
    code =  gen_code(llm, question, rel_col_docs, meta)    
    print(code)
    res = exec_code(code, df)    
    return {'question': question, 'result': res, 'hint_cols': [i.metadata['original_col_name'] for i in rel_col_docs]}

def execute_tests(llm, test_data, df, meta, cols_retriever):
    eval_res = []
    for test in test_data:
        t2 = test
        
        answer = pipeline(llm, test['question'], df, meta, cols_retriever)
        
        found_cols = []
        for expected_column in test['expected_columns']:
            if expected_column in answer['hint_cols']:
                found_cols.append(expected_column)
        t2['found_cols'] = found_cols
        
        found_cols_ratio = len(found_cols) / len(test['expected_columns'])
        t2['found_cols_ratio'] = found_cols_ratio

        t2['pipeline_result'] = answer['result']
        
        t2['hint_cols'] = answer['hint_cols']

        r = answer['result']
        
        if type(answer['result']) is tuple:
            print('finding float')
            for i in answer['result']:
                print(type(i))
                if type(i) is float or type(i) is float64:
                    print('found')
                    r = i
                    break
            
        t2['error'] = r - test['expected_answer']

        eval_res.append(t2)
        
    return eval_res