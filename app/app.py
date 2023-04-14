import sys
sys.path.append('..')

import os
import streamlit as st
from datetime import datetime
from modules.utilities import *
import pathlib
from uuid import uuid4


# Will need commenting when deploying the app
load_dotenv()

#Set env variables
setEnv()

aoai_embedding_model = 'text-search-davinci-doc-001' #'text-search-ada-doc-001'
aoai_embedding_model_version = '1'

aoai_text_model = 'text-davinci-003' 
aoai_text_model_version = '1'
aoai_text_model_temperature = 0.2
aoai_text_model_max_tokens = 500

aoai_embedding_model_deployment = aoai_embedding_models[aoai_embedding_model]["version"][aoai_embedding_model_version]["deployment_name"] #Azure OpenAI deployment name
aoai_embedding_model_dim = aoai_embedding_models[aoai_embedding_model]["version"][aoai_embedding_model_version]["dim"]

aoai_text_model_deployment = aoai_embedding_models[aoai_text_model]["version"][aoai_text_model_version]["deployment_name"] #Azure OpenAI deployment name

# top_n = 3 #How many answers to retrieve from index
score_threshold = 50 #Show answers above or equal to this score threshold
prompt_min_length = 5
ms_alias_min_length = 6
prompt_text_area_max_chars = 300
temp_dir = '../temp_uploads/' #Where uploaded files get staged until they are indexed, files staged for few seconds only then deleted.
app_version = '0.8' #Equal to docker image version tag, shown in sidebar.

#--------------------------------------------------------------------------
# Get connection
#--------------------------------------------------------------------------
az_redis = getRedisConnection(host=os.getenv('REDIS_HOST'), access_key=os.getenv('REDIS_ACCESS_KEY'), port=os.getenv('REDIS_PORT'), ssl=False)
# print(az_redis)

def getKeywordList(input_text):
    input_text = input_text.replace('.',' ')
    input_text = input_text.replace('-',' ')
    input_text = input_text.replace('=',' ')
    input_text = input_text.replace('?',' ')
    input_text = input_text.replace('!',' ')
    keyword_list = [word.lower() for word in input_text.split() if word.lower() not in ['?','a','an','and','or','do','of','if','not','for','are','was','were','is','can','have','has','there','their','the','how', 'why', 'when', 'what',"what's",'in', 'to', 'i', 'we', 'you']]
    return keyword_list

def highlightKeywords(keyword_list, input_text):
    highlighted = " ".join(f'<span style="background-color: #ffff99">{t}</span>' if t.lower() in keyword_list else t for t in input_text.split())
    return highlighted

# print(getKeywordList('What is HREC approval timeline?'))
# print(highlightKeywords(getKeywordList("What's weather?"),'Today is a great day as we have some good weather'))

def getResult(prompt, top_n, index_name):

    out = []

    # prompt = prompt + ' Respond with "Not found" if the answer is not present in the passage.'

    query_result,document_lc_list = queryRedis(az_redis_connection=az_redis, prompt=prompt, 
                            aoai_embedding_model=aoai_embedding_model_deployment, index_name=index_name, top_n=top_n)
    # print(f'query_result:{query_result}') 
    # print(f'document_lc_list:{document_lc_list}')

    # Check if any response received
    if document_lc_list is not None:

        # Open AI lc qna
        llm = AzureOpenAI(deployment_name=aoai_text_model_deployment,temperature=aoai_text_model_temperature, max_tokens=aoai_text_model_max_tokens)

        # lc
        # chain = load_qa_with_sources_chain(llm, chain_type="stuff")
        chain = load_qa_with_sources_chain(llm, chain_type="map_rerank", verbose=False, return_intermediate_steps=True)
        chain_out = chain({"input_documents": document_lc_list, "question": prompt}, return_only_outputs=False)
        # print(f'chain_out:{chain_out}')

        results = []
        for i, item in enumerate(chain_out['intermediate_steps']):
            # print(item['answer'], item['score']) #Uncomment to view the answer
            results.append((int(item['score']),i,item['answer']))

        results.sort(reverse = True) #Sort desc based on Score
        # print(results)
        # print(results[0][1]) #top first answer index    

        # Top N answers
        for i in range(top_n):
            
            # Uncomment to debug
            # print(f"\nAnswer: {results[i][2]}") #answer 
            # print(f"Score: {results[i][0]}") #answer score
            # print(bcolors.OKGREEN+f"Content {str(i+1)}: {chain_out['input_documents'][results[i][1]].page_content}"+bcolors.ENDC) #content
            # print(bcolors.BOLD+f"Source {str(i+1)}: {chain_out['input_documents'][results[i][1]].metadata['source']}"+bcolors.ENDC) #source
            # print(f"Similarity {str(i+1)}: {chain_out['input_documents'][results[i][1]].metadata['similarity']}") #similarity score
            # print(f"Page: {int(chain_out['input_documents'][results[i][1]].metadata['page'])+1}") #page

            # Check score threshold
            if int(results[i][0]) >= score_threshold:
                out_item = None
                out_item = {
                    "Answer":results[i][2],
                    "Score": int(results[i][0]),
                    f"Content": chain_out['input_documents'][results[i][1]].page_content,
                    f"Source": chain_out['input_documents'][results[i][1]].metadata['source'],
                    f"Similarity": chain_out['input_documents'][results[i][1]].metadata['similarity'],
                    f"Page": int(chain_out['input_documents'][results[i][1]].metadata['page'])+1
                    }
                out.append(out_item)        
    

    return out    

#--------------------------------------------------------------------------

# Initialization of session vars
if 'questions' not in st.session_state:
    st.session_state['questions'] = []
if 'answers' not in st.session_state:
    st.session_state['answers'] = []


st.set_page_config(page_title='Azure OpenAI Search Demo', layout='wide', page_icon='../images/logo_black_simple.png')



with st.container():
    
    def upload_button_click():

        if file_uploader is not None and len(textbox_msalias.strip()) >= ms_alias_min_length:
            progress_bar = middle_column_12.progress(0,'')
            
            # st.write(str(os.listdir('../')))
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

            # print(file_uploader.getvalue())
            # local_file = pathlib.Path('./temp_uploads/'+str(uuid4())+'_'+file_uploader.name)
            local_file = pathlib.Path(temp_dir + file_uploader.name)            
            local_file.write_bytes(file_uploader.getvalue()) #Write locally to crack open PDF/word docs
            
            local_file_path = str(local_file)
            # print(local_file_path)

            progress_bar.progress(20,'File acquired')

            # # Crack open PDF doc
            # document_pages_lc = readPDF(local_file_path)
            # print((document_pages_lc[0]))            

            progress_bar.progress(30,'Backend connected')

            # Create index if it does not exist
            result = createRedisIndex(az_redis_connection=az_redis, index_name=textbox_msalias , prefix = textbox_msalias, 
                                    distance_metric='COSINE', DIM = aoai_embedding_model_dim, vec_type='FLAT') 
            print(f'Create index result:{result}')

            progress_bar.progress(40,'Processing')

            # Read document, cleanse content, get content and embeddings
            document_page_content_list, \
            document_page_embedding_list, \
            document_page_no_list = getEmbeddingEntireDoc(documentPath=local_file_path, 
                                                        aoai_embedding_model=aoai_embedding_model_deployment, 
                                                        chunk_size=1)
            print('Embeddings retrieved')
            print(len(document_page_content_list), len(document_page_embedding_list), len(document_page_no_list))
            # print(document_page_content_list)
            # print(document_page_embedding_list, document_page_no_list)

            progress_bar.progress(80,'Almost done')

            # Add document pages
            response = addDocumentToRedis(az_redis_connection=az_redis, 
                            documentPath=local_file_path,
                            document_page_content_list=document_page_content_list, 
                            document_page_embedding_list=document_page_embedding_list, 
                            document_page_no_list=document_page_no_list,
                            prefix = textbox_msalias
                            )
            print(f'addDocumentToRedis: {response}')

            progress_bar.progress(90,'Running cleanup')

            # Remove local PDF after indexing completed
            if os.path.exists(local_file_path):
                os.remove(local_file_path)

            progress_bar.progress(100,'Completed')

        if len(textbox_msalias.strip()) < ms_alias_min_length:
            left_column.warning('Please enter a valid alias')

    top_left_column, middle_left_column, right_left_column = st.columns([40,20,40])
    top_left_column_1, top_left_column_2 = top_left_column.columns([20,80])
    top_left_column_1.image(image='../images/logo_black.png', width=100)
    # top_left_column_2.write('###')
    top_left_column_2.subheader('Semantic Search Demo')    
    top_left_column_2.write('Unleash the power of your documents with data-driven inquiries')    

    # st.write('---')   

    with st.sidebar:      
                    
        st.markdown(':gear: Settings')

        #Updated for general user
        textbox_msalias = st.text_input(label='Alias*', max_chars=10, key='textbox_msalias', type='password', value='demouser',label_visibility='visible', disabled=True)

        selectbox_top_n = st.selectbox(label='Top N results*',options=(1,2,3,4,5), index = 4, key='selectbox_top_n')        

        checkbox_score = st.checkbox(label='Score',key='checkbox_score', value=False, help='Value between 0 to 100 suggesting LLM confidence for answering the question by with retrieved passage of text.')
        checkbox_similarity = st.checkbox(label='Similarity',key='checkbox_similarity', value=False, help='Similarity between the query and retrieved passage of text.')   

        checkbox_page_no = st.checkbox(label='Page No',key='checkbox_page_no', value=True, help='Document page number.')    
        checkbox_show_fileupload = st.checkbox(label='Upload file',key='checkbox_show_fileupload', value=False, help='Upload file using upload widget.')

        st.write('### \n ### \n ### \n ### \n ### \n ### \n ### \n ### \n ### \n ### \n ### \n ### \n ### \n ### \n ### \n ### \n ### \n ### \n ###')
        st.caption('Version: '+app_version)
        st.write('<p style="font-size:14px; color:black;"><b>Powered by Azure OpenAI</b></p>', unsafe_allow_html=True)
    
    if checkbox_show_fileupload == True:
        st.write('----')

        left_column_11, middle_column_12, right_column_13 = st.columns([36,8,56])
        file_uploader = left_column_11.file_uploader(label='Upload file.',accept_multiple_files=False, key='file_uploader_1',type=['pdf', 'docx'],label_visibility='hidden')    
        middle_column_12.write('###')
        middle_column_12.write('###')
        # middle_column_b.write('###')    
        upload_button = middle_column_12.button(label='Upload', on_click=upload_button_click)       
        
        right_column_13.write('''<b><u>Disclaimer</u></b> 
                        \n <p style="font-size:16px; color:black;background-color:#fffce7">This demo app is not intended for use with sensitive data. 
                        We strongly advise against uploading any sensitive data to this application. 
                        We cannot guarantee the security of any data uploaded to this application. By using this application, you acknowledge that you understand and accept this risk.
                        Please use <b>publically available data</b> only.</p>''',unsafe_allow_html=True)
        st.write('----')       
        

with st.container():

    # left_column, middle_column, right_column = st.columns([46,8,46])
    left_column, middle_column, right_column = st.columns([60,8,32])
    
    prompt = left_column.text_area(label='Enter your question:',max_chars=prompt_text_area_max_chars, key='text_area1', label_visibility ='hidden')    

    def search_click():

        questions = st.session_state['questions']
        answers = st.session_state['answers']
        
        if prompt is not None and len(prompt.strip()) >= prompt_min_length and len(textbox_msalias.strip()) >= ms_alias_min_length:
            answer = []

            top_n = int(selectbox_top_n)

            try:
                answer = getResult(prompt, top_n, textbox_msalias)
            except:
                print('Exception in getResult()')

            #No results retrieved
            if len(answer)==0:
                left_column.warning('No results found. Consider uploading document/s first if you are using this app for the first time for alias you have specified. \n Show upload --> Browse file --> Click Upload to get started.')

            #Populate bottom pane with all N responses
            for ans_details in answer:

                keyword_list = getKeywordList(prompt)                

                left_column.write(f'<p style="font-size:16px; color:white;background-color:#7e93ff"><b>Answer</b>: {ans_details["Answer"]}</p>',unsafe_allow_html=True)                

                if checkbox_score:
                    left_column.write(f'<p style="font-size:12px; color:black"><b>Score</b>: {ans_details["Score"]}</p>',unsafe_allow_html=True)
                
                # left_column.write(f'<p style="font-size:16px; color:black;background-color:#e8ebfa""><b>Content</b>: {ans_details[f"Content"]}</p>',unsafe_allow_html=True)
                left_column.write(f'<p style="font-size:16px; color:black;background-color:#e8ebfa""><b>Content</b>: {highlightKeywords(keyword_list, ans_details[f"Content"])}</p>',unsafe_allow_html=True)                

                left_column.write(f'<p style="font-size:14px; color:black"><b>Source</b>:<i> {os.path.basename(ans_details[f"Source"])}</i></p>',unsafe_allow_html=True)
                
                if checkbox_similarity:
                    left_column.write(f'<p style="font-size:12px; color:black"><b>Similarity</b>: {ans_details[f"Similarity"]}</p>',unsafe_allow_html=True)
                
                if checkbox_page_no:
                    left_column.write(f'<p style="font-size:12px; color:black"><b>Page No</b>: {ans_details[f"Page"]}</p>',unsafe_allow_html=True)                
                
                left_column.write('----')

            if str(prompt).strip() != '' and len(answer) > 0:
                questions.append(prompt)
                answers.append(answer)        

            st.session_state['questions'] = questions          
            st.session_state['answers'] = answers 

        if len(textbox_msalias.strip()) < ms_alias_min_length:
            left_column.warning('Please enter a valid alias')

        # print(f'questions:{questions}')
        # print(f'answers:{answers}')

        if len(list(reversed(questions))) > 0:
            right_column.write(f'<p style="font-size:16px; color:black"><b>Question History</b></p>',unsafe_allow_html=True)    
            

        # Show in reversed order without modifying the lists set into sessions
        for i, item in enumerate(list(reversed(questions))):                       

            question_text = str('' + str(list(reversed(questions))[i]))
            # answer_text = str(''+ str(list(reversed(answers))[i]))
            
            # [{datetime.now().strftime("%d-%m-%Y %H:%M:%S")}]
            # right_column.write('###')
            
            right_column.write(f'<p style="font-size:16px; color:black;background-color:#f0f2f6"><b>Question</b>: {question_text} </p>',unsafe_allow_html=True)
            # right_column.write(f'<p style="font-size:14px; color:black"><b>Answer</b>: {answer_text}</p>',unsafe_allow_html=True)
            # right_column.write('---')
            # print(list(reversed(answers))[i])
            for j, ans_details in enumerate(list(reversed(answers))[i]):                
                
                #Only show 1 top answer in history (right side pane)
                if j==0:
                    right_column.write(f'<p style="font-size:14px; color:black"><b>Answer</b>: {ans_details["Answer"]}</p>',unsafe_allow_html=True)
                    if checkbox_score:
                        right_column.write(f'<p style="font-size:12px; color:black"><b>Score</b>: {ans_details["Score"]}</p>',unsafe_allow_html=True)
                    right_column.write(f'<p style="font-size:12px; color:black"><b>Content</b>: {ans_details[f"Content"]}</p>',unsafe_allow_html=True)
                    right_column.write(f'<p style="font-size:12px; color:black"><b>Source</b>:<i> {os.path.basename(ans_details[f"Source"])}</i></p>',unsafe_allow_html=True)
                    if checkbox_similarity:
                        right_column.write(f'<p style="font-size:12px; color:black"><b>Similarity</b>: {ans_details[f"Similarity"]}</p>',unsafe_allow_html=True)
                    if checkbox_page_no:
                        right_column.write(f'<p style="font-size:12px; color:black"><b>Page No</b>: {ans_details[f"Page"]}</p>',unsafe_allow_html=True)                
                    right_column.write('---')               


    def clear_click():
        st.session_state['text_area1'] = ''   
        st.session_state['questions'] = []
        st.session_state['answers'] = []
        # st.session_state['checkbox_score'] = False
        # st.session_state['checkbox_similarity'] = False
        # st.session_state['checkbox_page_no'] = False
    
       
    middle_column.write('###')
    middle_column.write('###')
    search_button= middle_column.button(label='Search', on_click= search_click)
    clear_button = middle_column.button(label='Clear', on_click = clear_click)    