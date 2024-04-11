from ydata_profiling import ProfileReport
import json
import pandas as pd
import base64
from IPython.display import Image, display
import matplotlib.pyplot as plt
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Pinecone
import pinecone
import streamlit as st
import numpy as np
from langchain.llms import AzureOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.vectorstores import Pinecone
import re
import base64
from IPython.display import Image, display
import matplotlib.pyplot as plt


config=json.load(
    open(
        r"config/config.json"
    )
)

def mm(graph):
    graphbytes = graph.encode("utf8")
    base64_bytes = base64.b64encode(graphbytes)
    base64_string = base64_bytes.decode("ascii")
    output = Image(url="https://mermaid.ink/img/" + base64_string)
    return output

def get_attribute_type(table_name,attribute_name,attribute_description,attribute_profile_report,llm):
    attribute_prompt = """You are an expert in identifying the type of a attribute in an entity.
    Given the attribute, find the type of the attribute(eg: fact columns, dimension columns, measure columns, description columns, key columns, date/time columns, flag columns ..etc.) based on its descriptions  and profile report

    Entity Name:{table_name}
    Attribute Name:{attribute_name}
    Attribute Description:{attribute_description}
    Attribute Profile Report:{attribute_profile_report}
    Example Output Format:
        Table Name:Customer
        Attribute Name: CustomerID
        Attribute Type: Dimension
        Reason:Detailed explantion for selecting the particular subject type based on input data 
    Give the output in the same format as example
    Only give the subheadings  sames as in EXAMPLE OUTPUT FORMAT


    """
    domain_template = attribute_prompt.format(
            table_name=table_name,
            attribute_name=attribute_name,
            attribute_description=attribute_description,
            attribute_profile_report=attribute_profile_report,
        )
    result = llm.predict(domain_template)
    lines = result.split("\n")
    attribute_type=lines[2].split(":")[1].strip()
    #st.write(attribute_type)
    return attribute_type


def get_domain_subject_area(table_name,attribute_details,llm,domain_file):
    domain_prompt = """You are an expert in deciding the correct and most suitable domain for an entity within a given for an entity within a given Retail CPG organization's data ecosystem from a list of input domain's file maintained by the organization.
    The entity name, attributes inside the table/entity and the attribute descriptions is given as input
    Input entity name:{table_name}
    Attributes inside the entity: {attribute_details}
    All the possible domains, their subject areas and the subject area descriptions are maintained by the organization is given in the domain config file:{domain_config}
    Your task is to map the entity based on the input entity and attribute details, domain config to the most suitable and appropriate Domain from the domain config
    EXAMPLE OUTPUT FORMAT:
        Entity_Name/Table Name:[Trade].Rpt_TradeRateReportEventDetail
        Domain: Sales
    NOTE:Make sure that the domain and subjet area within each suitable domain is selected only from the domain config file given as input by the organization properly.
    Give the output in the same format as in the EXAMPLE OUTPUT FORMAT.
    Seelct the domain only from the available domains
    The domain should be same as that in the input domain config file.""" 
    
    subject_area_prompt="""You are an expert in determing the subject area for an entity given iys attribute details and the domain details as input
    Input entity name:{table_name}
    Attributes inside the entity:{attribute_details}
    Domain of the input entity:{domain}
    All the possible subject area,subset area subset within {domain} domain maintained by the organization is given in the domain config file:{domain_config}
    Your task is to map the input entity to the most suitable subject area and the subject area subset within the input domain based on the entity attributes and descriptions
    EXAMPLE OUTPUT FORMAT:
        Entity_Name/Table Name:[Trade].Rpt_TradeRateReportEventDetail
        Domain: Sales
        Subject Area: Promotion
        Subject Area Subset(only if applicable): Customer_Demographics(optional else None)
        Reason:Detailed explantion for selecting the  particular domain,subject Area and subject Area  subset based on the input data 
    NOTE:Make sure that the subjet area within each suitable domain is selected only from the domain config file given as input by the organization properly.
    Give the output in the same format as in the EXAMPLE OUTPUT FORMAT.
    The subject area, subject area subset should be same as that in the input domain config file.
    Do not use any other terms from your knowlege while suggesting.
    Only give the subheadings  sames as in EXAMPLE OUTPUT FORMAT
    """  

    # domain_prompt = """You are an expert in deciding the most suitable domain, subdomain, subject level for an entity within a given Retail CPG organizations' data ecosystem from a list of input domain_config file maintained by the organization
    # The entity name, attributes inside the table/entity and their description for the attributes is given as input
    # Input entity name:{table_name}
    # Attributes inside the entity: {attribute_details}

    # Domain config file with the recognised domains,subject areas used in organization is given in a json:{domain_config}
    # The domain config file contains all the important domains , subject levels within each domain ,their descriptions and the subject area subset maintained by the organization.
    # Your task is to map the input entity to the most suitable domain, subject area and the subject area subset based on the entity attributes and descriptions
    # EXAMPLE OUTPUT FORMAT:
    #     Entity_Name/Table Name:Customer
    #     Domain: Sales
    #     Subject Area: Customer
    #     Subject Area Subset(only if applicable): Customer_Demographics(optional else None)
    #     Reason:Detailed explantion for selecting the  particular domain,subject Area and subject Area  subset based on the input data 
    # NOTE:Make sure that the domain and subjet area within each suitable domain is selected only from the domain config file given as input by the organization properly.
    # Give the output in the same format as in the EXAMPLE OUTPUT FORMAT
    # The domain,subject area should be same as that in the input domain config file.
    # Do not use any other terms from your knowlege while suggesting.
    # Only give the subheadings  sames as in EXAMPLE OUTPUT FORMAT
    # """
    domain_template = domain_prompt.format(
        table_name=table_name,
        attribute_details=attribute_details,
        domain_config=domain_file
    )
    result = llm(domain_template)
    lines = result.strip().split("\n")
    print("lines", lines)
    for idx, line in enumerate(lines):  # Iterate over lines with index
        if idx == 1:
            domain = line.split(":", 1)[1].strip()
    #st.write("domain",domain)
    subject_template=subject_area_prompt.format(
        table_name=table_name,
        domain=domain,
        attribute_details=attribute_details,
        domain_config=domain_file[domain_file["Domain"]==domain],
    )
    result1=llm(subject_template)
    #st.write("final result:",result1)



    return result1

def get_table_type(table_name, attribute_details, llm):
    domain_prompt = """You are an expert in deciding the most suitable type(fact,dimension, reference , lookup etc)for an entity within a given Retail CPG organizations' data ecosystem from a list of input domain_config file maintained by the organization
    The entity name, attributes inside the table/entity and their description for the attributes is given as input
    Input entity name:{table_name}
    Attributes inside the entity: {attribute_details}
    Select the table type (whether it is a fact,dimension, lookup,refernece etc) based on the input

    EXAMPLE OUTPUT FORMAT:
        Entity_Name/Table Name: Customer
        Table_Type: Dimension
        Reason:Detailed explantion for selecting the table type based on the input data 
    Give the output in the same format as in the EXAMPLE OUTPUT FORMAT
    Only give the subheadings  sames as in EXAMPLE OUTPUT FORMAT
    """
    domain_template = domain_prompt.format(
        table_name=table_name,
        attribute_details=attribute_details
    )
    result = llm(domain_template)
    return result


def get_granularity_data(table_name,domain,subject_area,subject_area_subset,table_type,attribute_list,hierarcial_info,llm):
    #main granularity prompt
    granularity_prompt = '''
    The entity name, possible primary keys, entity insights, domain, subject type, and all the dimension attributes names along with descriptions are provided as input.

    - Entity Name: {table_name}
    - Domain: {domain}
    - Subject Type: {Subject_Area}
    - Subject Type Subset: {Subject_Area_Subset}
    - Entity Type: {table_type}
    - Input Dimension List:{attribute_list}
    - Hierarchical Info: {hierarchical_info}
    From the provided hierarchical information:


    Give the least level of hierarchy for product and customer from the attribute list based on the ordering of hierarchies and levels specified in hierarchial info
    The least level of granularity should be only be selected from the input attribute list and  it should only consider the product based atrributes from input attribute list while selecting the lowest granulairty for product and
    it should only consider the customer based atrributes from input attribute list while selecting the lowest granulairty for customer
    Example Output:
            Input_Attribute_list:
                Conv
                Drug                 
                Grocery
                NUSA_CATEGORY_VALUE
                NUSA_SUB_CATEGORY_VALUE
                PL2_ID
                PL3_ID
                UPC_10_DIGIT
                UPC_12_DIGIT
                YEAR
            Lowest Level of Granularity for Product: UPC_10_DIGIT
            Lowest Level of Granularity for Customer: NA
            Lowest Level of Granularity other than Product and Customer: NA
    NOTE:The lowest granular column selected should be a attribute within the input attribute list
    Provide the output in the same format as the example.

    
    '''
    
    domain_template = granularity_prompt.format(
    table_name=table_name,
    attribute_list=attribute_list,
    domain=domain,
    Subject_Area=subject_area,
    Subject_Area_Subset=subject_area_subset,
    table_type=table_type,
    hierarchical_info=hierarcial_info,
    )
    result = llm(domain_template)
    #st.write("llm result",result)
    lowest_level_product = result.split("Lowest Level of Granularity for Product: ")[1].split("\n")[0]
    lowest_level_customer = result.split("Lowest Level of Granularity for Customer: ")[1].split("\n")[0]
    other_granularity= result.split("Lowest Level of Granularity other than Product and Customer: ")[1].split("\n")[0]
    print("lowest level product:",lowest_level_product )
    print("Lowest Level Customer:",lowest_level_customer)
    # Check if lowest level product matches related columns in the hierarchy
    main_product_entity_name = None
    for entity, data in hierarcial_info["Product Hierarchy"].items():
        if lowest_level_product in data["related_columns"]:
            main_product_entity_name = entity
            break

    # Check if lowest level customer matches related columns in the hierarchy
    main_customer_entity_name = None
    for entity, data in hierarcial_info["Customer_Hierarchy"].items():
        if lowest_level_customer in data["related_columns"]:
            main_customer_entity_name = entity
            break

    # Output main entity names for product and customer
    print("Main Entity Name for Product:", main_product_entity_name)
    print("Main Entity Name for Customer:", main_customer_entity_name)
    def get_higher_hierarchy(main_entity_name, hierarchy):
        #st.write("entered higher hierarchy function")
        higher_hierarchy = []
        found_entity = False
        for entity in hierarchy:
            if found_entity or entity == main_entity_name:
                found_entity = True
                higher_hierarchy.append(entity)
        #st.write("returned from hiray higher",higher_hierarchy)
        return higher_hierarchy

    
    higher_hierarchy_product = get_higher_hierarchy(main_product_entity_name, hierarcial_info["Product Hierarchy"])
    higher_hierarchy_customer = get_higher_hierarchy(main_customer_entity_name, hierarcial_info["Customer_Hierarchy"])

    print("Higher Hierarchy for Product:", higher_hierarchy_product)
    print("Higher Hierarchy for Customer:", higher_hierarchy_customer)
    merged_hierarchy = higher_hierarchy_product + higher_hierarchy_customer
    if not merged_hierarchy:
    # Check if lowest_level_product is not "NA"
        if lowest_level_product != "NA":
            merged_hierarchy.append(lowest_level_product)
        if lowest_level_customer != "NA":
            merged_hierarchy.append(lowest_level_customer)
        if other_granularity != "NA":
            merged_hierarchy.append(other_granularity)
    #st.write("merged",merged_hierarchy)
    return merged_hierarchy

def get_purpose_of_calculation(domain,subject_area,measure,description,profile,llm):
    metric_prompt='''You are an expert in finding the possible calculation method and the purpose of a measure.
    Domain:{domain}
    Subject Area:{subject_area}
    Input Measure:{measure}
    Input Measure Description:{description}
    Input Measure Profile report:{profile}
    The input contains the measure name along with its domain, subject area,description and the profile report
    Your task to find the the possible calculation method and the main purpose of the measure based on the input details
    Example Output Format:
        Input Measure:Total_Sales
        Possible Calculation:TotalSales=Sum of all individual sales transactions
        Purpose of the measure:Quantify the total revenue generated from all customer transactions across various products and services within the Sales domain.
    Give the output as in the same format of the example
    '''

    metric_template=metric_prompt.format(measure=measure,description=description,profile=profile,domain=domain,subject_area=subject_area)
    result=llm(metric_template)
    possible_calculation_method = result.split("Possible Calculation:")[1].split("\n")[0]
    purpose_of_measure = result.split("Purpose of the measure:")[1]
    return possible_calculation_method ,purpose_of_measure



def insert_data_into_pinecone(attribute_data_trunc):
    index_name = "llm-accelerator-openai"
    MODEL = "text-embedding-ada-002"
    embeddings = OpenAIEmbeddings(
    deployment="llm-acelerator-embedding",
    model=MODEL,
    )

# initialize connection to pinecone
    pinecone.init(
        api_key="6bc62698-733d-481f-af2d-b1cb8bd61312", environment="us-west4-gcp-free"
    )
    measure_dict = {}
    all_measures_data = []
    domain=attribute_data_trunc["Domain"].unique()[0]
    subject_area=attribute_data_trunc["Subject_Area"].unique()[0]
    attribute_data_trunc=attribute_data_trunc[attribute_data_trunc["Attribute Type"]=='Measure']
    for index, row in attribute_data_trunc.iterrows():
        measure_dict = {}
        measure_name = row["Attribute Name"]
        measure_description = row["Attribute Description"]
        measure_dict[measure_name] = {"description": measure_description}
        measure_profile = row["Attribute Profile Report"]
        measure_granularity=row["Granularity Level"]
        measure_purpose=row["Purpose Of Measure"]
        measure_calculation=row["Possible Method Of Calculation"]
        measure_dict[measure_name] = {
            "Domain": domain,
            "Subject Area": subject_area,
            "Description": measure_description,
            "Purpose Of Measure":measure_purpose,
            "Possible Method Of Calculation":measure_calculation,
            "Granularity Level": measure_granularity,
            "Profile_Report": measure_profile,
        }
        all_measures_data.append(measure_dict)
    #st.write(len(all_measures_data))
    #st.write("all measures data",all_measures_data)
    namespace = f"{domain}_{subject_area}"
    print("name", namespace)
    #st.write(type(all_measures_data))
    logs = []

    for item in all_measures_data:
        log_entry = {}
        for key, value in item.items():
            try:
                json.dumps({key: value})
                log_entry[key] = value
            except TypeError:
                if isinstance(value, np.ndarray):
                    log_entry[key] = value.tolist()
                else:
                    log_entry[key] = str(value)

        logs.append(json.dumps(log_entry))
    #st.write("logs:",logs)
    #st.write("len of logs:",len(logs))

    vector_log = Pinecone.from_texts(
        logs,
        embeddings,
        index_name=index_name,
        namespace=namespace,
        metadatas=[{"source": f"logs{i}"} for i in range(0, len(logs))],
    )
    st.write(f"Data Inserted into {namespace} in vector db")

def get_duplicates(domain,subject_area,entity_domain_data,llm):
    index_name = "llm-accelerator-openai"
    pinecone.init(
        api_key="6bc62698-733d-481f-af2d-b1cb8bd61312", environment="us-west4-gcp-free"
    )
    MODEL = "text-embedding-ada-002"
    embeddings = OpenAIEmbeddings(
    deployment="llm-acelerator-embedding",
    model=MODEL,
    )
    #st.write(entity_domain_data)
    #st.write("hello")
    namespace = f"{domain}_{subject_area}"
    #st.write("Namespace:", namespace)
    resolution_metadata = Pinecone.from_existing_index(
        index_name, embeddings, namespace=namespace
    )
    #st.write(resolution_metadata)
    domain_subject_dict = {"Domain": domain, "Subject Area": subject_area, "Duplicate Attributes": {}}
    attribute_data_trunc = entity_domain_data[(entity_domain_data["Domain"] == domain)&(entity_domain_data["Subject_Area"] == subject_area)&(entity_domain_data["Attribute Type"] == "Measure")]
    #st.write(attribute_data_trunc)
    if not attribute_data_trunc.empty:
        measures_list=[]
        for index, row in attribute_data_trunc.iterrows():
            #st.write("index",index)
            #st.write("row",row)
            measure_details=[]
            input_var = None
            check_dup = []
            measure_dict = {}
            measure_name = row["Attribute Name"]
            measure_table_name=row["Table Name"]
            measure_description = row["Attribute Description"]
            measure_dict[measure_name] = {"description": measure_description}
            measure_profile = row["Attribute Profile Report"]
            measure_granularity=row["Granularity Level"]
            measure_purpose=row["Purpose Of Measure"]
            measure_calculation=row["Possible Method Of Calculation"]
            measure_dict[measure_name] = {
                "Table Name":measure_table_name,
                "Domain": domain,
                "Subject Area": subject_area,
                "Description": measure_description,
                "Purpose of the measure":measure_purpose,
                "Possible Calculation Method":measure_calculation,
                "Granularity": measure_granularity,
                "Profile_Report": measure_profile,
            }
            measure_dict=json.dumps(measure_dict)
            #st.write("measure_dict",measure_dict)
            similar_data=resolution_metadata.similarity_search(measure_dict)
            #st.write("Vector search result:",similar_data)
            
            for doc in similar_data:
                #st.write("doc",doc)
                page_content = doc.page_content
                try:
                    content_dict = json.loads(page_content)
                    column_name = next(iter(content_dict))
                    if measure_name == column_name:
                        input_var = doc.page_content
                        
                    else:
                        check_dup.append(doc.page_content)
                        #st.write("check_dup",check_dup)
                except Exception as e:
                    print(f"Error processing document: {e}")
            #st.write("Input Attribute Details:",input_var)
            #("Possible Duplicate List:",check_dup)
            #st.write("duplicates",check_dup)
            dup_prompt = f"""You are an expert in finding out the true duplicates of a measure
            The input is a {domain} domain {subject_area} subject area measure used in a CPG retail organization.
            Input Measure:{measure_name}
            Input measure details :{input_var}
            Possible list of Duplicate measures of the input:{check_dup}
            You are provided with a possible list of duplicate measures for the input measure. Your objective is to carefully compare the input attribute with each potential duplicate and filter out the true duplicates using a systematic approach.
            Follow these steps to filter out the duplicates from the possible dupliate list of measures:
                1. Begin by identifying duplicate measures with exactly same purpose or notion to the input measure
                2. Next, from the identified set,filter the set of duplicates  based on common or similar  possible method of calculation to the input measure.
                3. Then from the filtered set, consider filtering out the duplicates with similarities in data uniqueness,data distributions ,data patterns and other significant attributes from the profile report to the input measure.
                4. Lastly, refine the finally recieved set of selection based on similar descriptions and granularities to the input measure
            Now give the final set of the finally filtered out and the actual duplicates to the input measure
            Be carefull while choosing the duplicate attributes as the chosen duplicates are the original duplicates of the given input measure and will be removed from the actual table.
            Example Output Format:
                INPUT MEASURE: Actual Spend
                POSSIBLE LIST OF DUPLICATE MEASURES(from input):
                    -AggregateSales
                    -TotalTrendSpend
                    -GrossRevenue"
                    -TotalNonExcludedAmount
                ACTUAL DUPLICATES:
                    -TotalTrendSpend
                    -TotalNonExcludedAmount
            The output should contain the true duplicates only from the possible duplicates list
            Give the output only with 3 subheadings - INPUT MEASURES, POSSIBLE DUPLICATES,ACTUAL DUPLICATES
            DO not add any other details in the output.
            If no duplicates are found, return "None" in the "ACTUAL DUPLICATES" section.
            """ 
            result=llm(dup_prompt)
            #st.write("result:",result)
            actual_duplicates_section = re.search(r'ACTUAL DUPLICATES:\n(.+?)(?=\n\n|$)', result, re.DOTALL)


            if actual_duplicates_section:
                actual_duplicates = actual_duplicates_section.group(1).strip().replace("-","").split("\n")
                print(actual_duplicates)
                domain_subject_dict["Duplicate Attributes"][measure_name] = [dup.strip() for dup in actual_duplicates]
            else:
                print("No 'ACTUAL DUPLICATES' section found.")
            #st.write("final list",domain_subject_dict)
            #st.write("--------------------------------------------------")
        measures_list.append(domain_subject_dict)
        def clean_measure_dict(measures):
            # Identify keys with "None" values
            none_keys = [key for key, value in measures.items() if "None" in value]

            # Remove keys with "None" values
            for none_key in none_keys:
                del measures[none_key]

            # Remove keys with "None" values from other keys' values
            for key, value in measures.items():
                measures[key] = [v for v in value if v not in none_keys]

            # Remove keys with empty lists as values
            empty_keys = [key for key, value in measures.items() if not value]
            for empty_key in empty_keys:
                del measures[empty_key]

            return measures

        # Update the measures_list to contain the cleaned duplicate attributes
        for domain_subject_dict in measures_list:
            duplicate_attributes = domain_subject_dict['Duplicate Attributes']
            cleaned_duplicate_attributes = clean_measure_dict(duplicate_attributes)
            domain_subject_dict['Duplicate Attributes'] = cleaned_duplicate_attributes

        # Print the updated measures_list
        #st.write("updated",measures_list)
        # Function to find final list of duplicates within a dictionary of attributes
        def find_duplicates(relations):
            duplicates = []
            for item, duplicates_list in relations.items():
                for duplicate in duplicates_list:
                    if item in relations.get(duplicate, []) and (duplicate, item) not in duplicates:
                        duplicates.append((item, duplicate))
            return duplicates

        # Initialize an empty list to store the results
        duplicates_list = []

        # Iterate through the measures_list
        for domain_subject_dict in measures_list:
            domain = domain_subject_dict['Domain']
            subject_area = domain_subject_dict['Subject Area']
            duplicate_attributes = domain_subject_dict['Duplicate Attributes']
            
            # Find duplicates within the duplicate attributes dictionary
            duplicates = find_duplicates(duplicate_attributes)
            
            # Append the duplicates to the list
            duplicates_list.append({
                'Domain': domain,
                'Subject Area': subject_area,
                'Duplicate_Attributes': [[pair[0], pair[1]] for pair in duplicates]
            })

        # Print the list of dictionaries
        #st.write("final",duplicates_list)
        for item in duplicates_list:
            domain = item['Domain']
            subject_area = item['Subject Area']
            duplicate_attributes = item['Duplicate_Attributes']
            filtered_data = entity_domain_data[(entity_domain_data['Domain'] == domain) & (entity_domain_data['Subject_Area'] == subject_area)]
            for attr_pair in duplicate_attributes:
                table0=entity_domain_data[entity_domain_data["Attribute Name"]==attr_pair[0]]["Table Name"].unique()[0]
                table1=entity_domain_data[entity_domain_data["Attribute Name"]==attr_pair[1]]["Table Name"].unique()[0]
                granularity1=entity_domain_data[entity_domain_data["Attribute Name"]==attr_pair[1]]["Granularity Level"].unique()[0]
                granularity0=entity_domain_data[entity_domain_data["Attribute Name"]==attr_pair[0]]["Granularity Level"].unique()[0]
                st.markdown(f"**{attr_pair[0]}** of `{table0}` with granularity `{granularity0}` is a duplicate of **{attr_pair[1]}** of `{table1}` with `{granularity1}`")
                    
def generate_mermaid_code(hierarchy_config,value):
    mermaid_code = "graph TD;\n"
    levels = hierarchy_config[f"{value}"]
    keys = list(levels.keys())
    
    # Iterate over keys in reverse order
    for i in range(len(keys) - 1, 0, -1):
        current_key = keys[i]
        current_level = levels[current_key]
        prev_key = keys[i-1]
        prev_level = levels[prev_key]
        mermaid_code += f"    {current_key}[{current_key}:{current_level['hierarchical_level_category']}] --> {prev_key}[{prev_key}: {prev_level['hierarchical_level_category']}]\n"

    img=mm(mermaid_code)
    display(img)
    return img      
             
                