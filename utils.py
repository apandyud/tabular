from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.document_transformers import EmbeddingsRedundantFilter
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from langchain_community.cache import SQLiteCache


import numpy as np

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
    prompt1 = f"Please list the typical database column fields, that required to answer the following question. Just the list items, separated by comma, do not write explanation. Question: {question}"
    relevant_col_list_msg = llm.invoke(prompt1)
    relevant_col_list = relevant_col_list_msg.content
    return relevant_col_list


def match_column_names(hints_text, retriever):
    #filter = EmbeddingsRedundantFilter(embeddings=OpenAIEmbeddings(), similarity_threshold=0.99)
    res = []
    hints = hints_text.split(',')
    for hint in hints:
        print('hint',hint)
        rel_col_docs = retriever.invoke(hint)    
        #print([i.metadata['original_col_name']  for i in rel_col_docs])
        #rel_col_docs = filter.transform_documents(rel_col_docs)
        print([i.metadata['original_col_name']  for i in rel_col_docs])
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
#    prompt2 = f"Given a dataframe with the following columns {columns}, column meaning: {data_explanation}, can you generate a python code, without sample data, which can answer the following question? the code must contain only one function called 'run', that returns an exact number of type 'float'. Use dropna function on used columns. Do not write explanation, just code.\nQuestion: {question}"
    prompt2 = f"Given a dataframe with the following columns {columns}, column meaning: {data_explanation}, can you generate a python code, without sample data, which can answer the following question? the code must contain only one function called 'run', and no wrapping class. The function would return the numeric results and used statistical method name as well in format (result, statistical_). Use dropna function on used columns. Do not write explanation, just code.\nQuestion: {question}"
    res = llm.invoke(prompt2)
    print(res)
    code = res.content.replace('```python','').replace('```','')
    return code
    
def explore_columns(code, rel_col_docs ):
    columns = [i.metadata['original_col_name'] for i in rel_col_docs]    
    res = []
    for column in columns:
        if column in code:
            res.append(column)    
    return res

def avgs(l):
    if isinstance(l[0] , (int, float)):
        return sum(l) / len(l)
    if isinstance(l[0], np.ndarray):
        return np.average(l, axis = 0)        
    if isinstance(l[0], tuple):
        #return np.average(l, axis = 0)
        numeric_averages = {}
        for tuple_obj in l:
            for idx, value in enumerate(tuple_obj):
                if isinstance(value, (int, float)):                    
                    numeric_averages[idx] = numeric_averages.get(idx, 0) + value        
        r = []
        for i in range (len(l[0])):            
            if i in numeric_averages:                
                r.append(numeric_averages[i] / len(l))
            else:
                
                r.append(l[0][i])
        return tuple(r)
    if isinstance(l[0], dict):
        objects = l
        numeric_averages = {}
        for obj in l:
            for key, value in obj.items():
                if isinstance(value, (int, float)):
                    numeric_averages[key] = numeric_averages.get(key, 0) + value
        
        for key, total in numeric_averages.items():
            count = len([obj for obj in objects if isinstance(obj.get(key), (int, float)) or (isinstance(obj, list) or isinstance(obj, np.ndarray) and isinstance(value, (int, float)))])
            numeric_averages[key] = total / count
        return numeric_averages

def exec_code(code, df, used_columns):    
    if 'STUD_MATH' in used_columns:
        print('Iterating "STUD_MATH" ')
        res = []
        for i in range(1, 6):
            df['STUD_MATH'] = df['PV'+str(i)+'MATH']
            df2 = df[used_columns]                    
            loc = locals()    
            exec(code + "\nr = run(df2)\n", globals(), loc)
            res.append(loc['r'])            
        print(res)
        return avgs(res)
    else:
        exec(code + "\nr = run(df)\n", globals(), loc)
        return loc['r']
def interpret(llm, question, result):
    prompt = f"Given the following question an result, please interpret the results. Question: {question} Result: {result}"
    inter = llm.invoke(prompt).content
    return inter
    
def pipeline(llm, question, df, meta, col_retriever):
    col_hints = generate_column_name_hints(llm, question)
    print(col_hints)
    rel_col_docs = match_column_names(col_hints, col_retriever)    
    #print([i.page_content for i in rel_col_docs])
    code =  gen_code(llm, question, rel_col_docs, meta)    
    print(code)
    used_columns = explore_columns(code, rel_col_docs )
    print(used_columns)
    #columns = [i.metadata['original_col_name'] for i in rel_col_docs]
    #df2 = df.dropna(subset=columns)
    res = exec_code(code, df, used_columns)
    inter =  interpret(llm, question, res)
    return {'question': question, 'result': res, 'inter': inter, 'used_columns': used_columns, 'hint_cols': [(i.metadata['original_col_name'], i.page_content) for i in rel_col_docs]}

def pipeline_l2(llm, question,  datasets):
    dsname = llm.invoke('Regarding the following question, is it about schools, students, or none of them? Please answer "SCHOOL", "STUDENT" or "NONE" according to the quetsion. ' + question).content
    if dsname == "NONE":
        return {'question': question, 'result': None, 'hint_cols': []}
    
    return pipeline(llm, question, datasets[dsname].df, datasets[dsname].meta, datasets[dsname].col_retriever)
    
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