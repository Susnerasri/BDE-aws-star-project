import streamlit as st
import pandas as pd
import os
import argparse
import extra_streamlit_components as stx
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import AzureOpenAI
from pandas import DataFrame
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from utils.metadata_curator import (
    profile_datasets,
    get_description_and_mermaid_code,
    restructure_profile_report_for_analysing,
    get_matching_attribute,
    mm
)
from utils.deduplication import (
    get_attribute_type,
    get_domain_subject_area,
    get_table_type,
    get_granularity_data,
    get_purpose_of_calculation,
    insert_data_into_pinecone,
    get_duplicates,
    generate_mermaid_code,
    mm
)
from langchain.llms import VertexAI
from langchain.embeddings import VertexAIEmbeddings
import json
from ydata_profiling import ProfileReport
import base64
from IPython.display import Image
from io import StringIO
import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus
import mysql.connector







st.set_page_config(
    "Generative AI Accelerator", layout="wide", initial_sidebar_state="collapsed"
)
no_sidebar_style = """
<style>
    [data-testid="collapsedControl"] {
        display: none
    }
    div.block-container.css-z5fcl4.egzxvld4{
    padding-top:0px;
    }
</style>
"""
horizontal_radio_button = """
<style>
div.row-widget.stRadio > div{flex-direction:row;}
</style>
"""
hide_bar = """
<style>
div.css-1bcsrn5.e1tzin5v1 {
        display:none;
    }
</style>
"""
hide_homepage = """
<style>
div.css-j7qwjs.e1fqkh3o7
{
display:none;}
</style>
"""
st.markdown(horizontal_radio_button, unsafe_allow_html=True)
st.markdown(hide_homepage, unsafe_allow_html=True)

st.markdown(
    """
<style>
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {

}

    div.css-18fuwiq.e1tzin5v1{
     overflow-y:scroll;
        height:500px;
        max-height:500px;
        width:13px;
        top:10%;
        bottom:10%;
        opacity:100%;
        z-index:2;
        box-sizing: border-box;
        border-radius: 1px;
        border-style: solid;
        border-color:grey;
        border-width:1px;
        padding:10px;
    }
</style>
""",
    unsafe_allow_html=True,
)


st.markdown(
    """<style>
    div.css-1enkj8y.e1tzin5v1{
    display:None;
}
""",
    unsafe_allow_html=True,
)

path_obj = json.load(
    open(
        r"config/paths.json"
    )
)
config=json.load(
    open(
        r"config/config.json"
    )
)

mydb = mysql.connector.connect(
    host=config["MYSQL"]["host"], user=config["MYSQL"]["user"], password=config["MYSQL"]["password"], database="data_resolution1"
)
engine = create_engine(
                        "mysql+mysqlconnector://root:%s@localhost/data_resolution1"
                        % quote_plus("Root@123")
                    )
mycursor = mydb.cursor()


# Parse the subscription type based on command line argument
parser = argparse.ArgumentParser()
#st.write(parser)
parser.add_argument(
    "--subscription", default="OpenAI", type=str, help="Enter the subscription type"
)
opt = parser.parse_args()
subscription_type = opt.subscription
st.session_state["subscription_type"] = subscription_type
#st.write(st.session_state["subscription_type"])
success_placeholder = st.empty()



class standardize_names:

    def __init__(self):
        f = open(
            r"config/config.json"
        )
        config = json.load(f)
        f.close()
        os.environ["OPENAI_API_TYPE"] = config["OpenAI"]["OPENAI_API_TYPE"]
        os.environ["OPENAI_API_VERSION"] = config["OpenAI"]["OPENAI_API_VERSION"]
        os.environ["OPENAI_API_BASE"] = config["OpenAI"]["OPENAI_API_BASE"]
        os.environ["OPENAI_API_KEY"] = config["OpenAI"]["OPENAI_API_KEY"]
        llm=st.session_state["llm"]
        # self.llm = AzureOpenAI(
        #     engine=config["OpenAI"]["engine"],
        #     model_name=config["OpenAI"]["model_name"],
        #     temperature=config["OpenAI"]["temperature"],
        # )
        self.llm=llm

    def match_standardized_name(self, standardized_names_df, col_name):
        try:
            standardized_name = standardized_names_df[
                standardized_names_df["Other Business Names"] == col_name
            ]["Standardized Name"].iloc[0]
        except:
            standardized_name = ""
        return standardized_name

    def match_data_dictionary(self, data_dictionary_df, col_name):
        try:
            data_dict_match = data_dictionary_df[
                data_dictionary_df["Column"] == col_name
            ]["Description"].iloc[0]
        except:
            data_dict_match = ""
        # st.write("data dict inside match dict:",data_dict_match)
        return data_dict_match

    def data_domain_mapper(self, data_domain_df, col_name):
        data_domain_dict = dict(
            zip(data_domain_df["Data Domain"], data_domain_df["Class Word"])
        )
        for key in data_domain_dict.keys():
            if key.lower() == col_name.lower():
                return key, data_domain_dict[key]
            if data_domain_dict[key].lower() == col_name.lower():
                return key, data_domain_dict[key]
        return "none", "none"

    def check_data_domain_presence(self, data_domain_df, col_name):
        data_domain_dict = dict(
            zip(data_domain_df["Data Domain"], data_domain_df["Class Word"])
        )
        # st.write("data_dict within check_data_domain_presence",data_domain_dict )
        for key in data_domain_dict.keys():
            try:
                # st.write("key",key)
                # st.write("col name",col_name)
                # st.write("len of colname index without key:",col_name.lower().rindex(key.lower()))
                # st.write("len of col name index",col_name.lower().rindex(key.lower()) + len(key))
                # st.write("len of col ", len(col_name))
                if col_name.lower().rindex(key.lower()) + len(key) == len(col_name):
                    # st.write("entered if after index length check")

                    return key, data_domain_dict[key]
            except:
                pass
            try:
                if col_name.lower().rindex(data_domain_dict[key].lower()) + len(
                    data_domain_dict[key]
                ) == len(col_name):
                    return key, data_domain_dict[key]
            except:
                pass
        return "none", "none"

    def match_data_domain(
        self, data_domain_df, source_col, target_col, data_dict_match
    ):
        # st.write("entered match data domain")
        # st.write("data dictionary inside match data domain:",data_dict_match)
        domain = "none"
        tgt_col_list = target_col.split(" ")
        # st.write("tgt_col_list1:",tgt_col_list)
        if len(tgt_col_list) > 1:
            # st.write("entered if")
            domain, classword = self.data_domain_mapper(
                data_domain_df, tgt_col_list[-1]
            )
            # st.write("domain:",domain)
            # st.write("classword",classword)
        else:
            # st.write("entered elif")
            tgt_col_list = target_col.split("_")
            if len(tgt_col_list) > 1:
                domain, classword = self.data_domain_mapper(
                    data_domain_df, tgt_col_list[-1]
                )
            else:
                # st.write("print else within elseif")
                domain, classword = self.check_data_domain_presence(
                    data_domain_df, target_col
                )
        if domain == "none":
            # st.write(target_col)
            # st.write("entered if domain is equal to none")
            match_domain_prompt = f"""You are a data expert in finding out the data domain of an attribute by matching the attribute description with the data domain description. The data domain should be one from the below given list

        Data domain list:
        {data_domain_df}

        Source column name:
        {source_col}

        Modified column name:
        {target_col}

        Appropriate Description:
        {data_dict_match}

        Return the data domain along with the classword to which the attribute may belong to

        Output format:
        {{data_domain}},{{classword}}

        NOTE:Return the output only as in the Output format.
        DO not give any other details or information in the output
        Give only the data_domain and classword as in output format


        """
            op = self.llm(match_domain_prompt)
            # st.write("llm_output:",op)
            domain = op.split(",")[0]
            # st.write("domain:",domain)
            classword = op.split(",")[1]
            # st.write("classword:",classword)
        return domain, classword

    def create_standardized_name(
        self, target_col, matched_data_domain, data_dict_match
    ):
        create_standardized_name_prompt = f"""You are an expert data modeller. You will be given an attribute name along with its description. Your job is to generate a name that well describes the target attribute based on attribute description within 65 characters.

        Attribute name:
        {target_col}
        Attribute Description:
        {data_dict_match}

        Standardized attribute name:

        Rules to follow while generating standardized name:
            1)Try to generate the name adapting from the current attribute name
            2)The generated name should not have any underscores(_)
            3)Do not use short form of words while generating the name. Always use full word 
            4)Number of characters in the generated standardized name SHOULD ALWAYS BE LESS than 65 characters

        Note:
        Just return the name. Do not give explanation
"""
        standardized_name = self.llm(create_standardized_name_prompt)
        return standardized_name

    def replace_classword_with_data_domain(self, target_col, domain, classword):
        replace_classword_with_data_domain_prompt = f"""You are an expert data modeller. You will be given an attribute name. You will be given a data domain and the equivalent classword. If you find the classword in the target column replace it with the data domain. Else just return the input column name as output
        Attribute name:
        {target_col}
        Data domain:
        {domain}
        Classword:
        {classword}

        Attribute name:

        Note:
        Just return the final corrected attribute name. Do not give explanations
        """
        return self.llm(replace_classword_with_data_domain_prompt)

    def append_data_domain_name(self, target_col, matched_data_domain):
        try:
            if target_col.lower().index(matched_data_domain.lower()) + len(
                matched_data_domain
            ) == len(target_col):
                return target_col
            else:
                return target_col + matched_data_domain
        except:
            return target_col + matched_data_domain

    #         append_data_domain = f"""You are a expert data modeller. Your job is to check whether the target attribute name is having the data domain name at the last. If it is present return the same attribute name. If the data domain is not present at the last, the rename to column in a meaningfull way such that the data domain name is at the last

    #         Attribute name:
    #         {target_col}
    #         Data domain:
    #         {matched_data_domain}

    #         Attribute name:

    #         Note:
    #         Just return the final corrected attribute name. Do not give explanations
    # """
    #         return self.llm(append_data_domain)

    def pascal_case_check(self, target_col):
        pascal_check_prompt = f"""You will be given a word for which you have to given the Pascal case way of writing it.
        Name:
        {target_col}
        Pascal Case name:
        """
        return self.llm(pascal_check_prompt)

    def generate_dq_rules():
        pass


def mm(graph):
    graphbytes = graph.encode("utf8")
    base64_bytes = base64.b64encode(graphbytes)
    base64_string = base64_bytes.decode("ascii")
    output = Image(url="https://mermaid.ink/img/" + base64_string)
    return output


def success_notification(msg):
    if "LLM" in msg:
        success_placeholder.success("LLM Details are updated successfully")


with st.sidebar:
    st.header("Update the LLM Details")
    update_llm = st.radio(
        "Enter the details of LLM",
        ["Hide", "LLM Details"],
    )
    if update_llm == "LLM Details":
        st.header("Enter LLM Details")
        if subscription_type == "OpenAI":
            st.subheader("Enter OpenAI details")
            api_key = st.text_input("Openai access key", type="password")
            model = st.selectbox("Model Name", ("None", "gpt-3.5-turbo"))
            st.session_state["model"] = model
            st.session_state["api_key"] = api_key
        elif subscription_type == "GCP":
            st.subheader("Enter GCP details")
            model = st.selectbox(
                "Model Name", ("None", "text-bison", "code-bison", "gemini-pro")
            )
            llm_accessibility = st.selectbox(
                "Access llm via", ("None", "IAM Role", "Access key")
            )
            if llm_accessibility == "Access key":
                api_key = st.text_input("Palm API", type="password")
                st.session_state["api_key"] = api_key
            st.session_state["model"] = model

        elif subscription_type == "Azure":
            st.subheader("Enter Azure OpenAI details")
            api_type = st.text_input("Enter Api type")
            api_version = st.text_input("Enter Api version")
            api_base = st.text_input("Enter Api base")
            api_key = st.text_input("Api key", type="password")
            deployment_name = st.text_input("Enter model deployment name")
            model = st.selectbox("Model Name", ("None", "gpt-3.5-turbo"))
            st.session_state["api_type"] = api_type
            st.session_state["api_version"] = api_version
            st.session_state["api_base"] = api_base
            st.session_state["api_key"] = api_key
            st.session_state["deployment_name"] = deployment_name
            st.session_state["model"] = model
        submit = st.button("submit")
        if submit:
            try:
                if subscription_type == "GCP":
                    llm = VertexAI(
                        model_name=model,
                        allowed_model_args=["temperature", "max_output_tokens"],
                        max_output_tokens=1000,
                        temperature=0.1,
                    )
                    st.session_state["llm_chat"] = llm
                    print(llm)
                    embeddings = VertexAIEmbeddings()
                    st.session_state["embeddings"] = embeddings
                elif subscription_type == "Azure":
                    os.environ["OPENAI_API_TYPE"] = api_type
                    os.environ["OPENAI_API_VERSION"] = api_version
                    os.environ["OPENAI_API_BASE"] = api_base
                    os.environ["OPENAI_API_KEY"] = api_key
                    llm = AzureOpenAI(
                        engine=deployment_name, model_name=model, temperature=0
                    )
                    llm_chat = AzureChatOpenAI(
                        deployment_name=deployment_name, temperature=0
                    )
                    MODEL = "text-embedding-ada-002"
                    embeddings = OpenAIEmbeddings(
                        deployment="llm-acelerator-embedding",
                        model=MODEL,
                        openai_api_base=st.session_state["api_base"],
                        openai_api_version=st.session_state["api_version"],
                        openai_api_key=st.session_state["api_key"],
                        openai_api_type=st.session_state["api_type"],
                    )
                    st.session_state["embeddings"] = embeddings
                    st.session_state["llm_chat"] = llm_chat
                st.session_state["llm"] = llm

                success_notification("LLM details are updated successfully")
            except:
                success_notification("LLM details are invalid")


st.header("Standardization of Data Model")


def create_matching_dict(match_dict, old_name, standardized_name):
    match_dict[old_name] = standardized_name
    return match_dict


datamodel_subtab = stx.tab_bar(
    data=[
        stx.TabBarItemData(id="Source Analyze", title="Analyze Source", description=""),
        stx.TabBarItemData(
            id="Standardize Data Model",
            title="Standardize Data Model",
            description="",
        ),
        stx.TabBarItemData(id="Deduplication", title="Deduplication", description=""),
    ]
)
if datamodel_subtab == "Source Analyze":

    st.session_state["output_pyspark"] = None
    st.session_state["llm_feature"] = "Source Analyze"
    st.subheader("Pick up your Data")
    source = st.selectbox(
        "Select data scource...",
        ("Browse from Local", "AWS S3", "GCP GCS", "Azure Blob"),
    )
    if source == "Browse from Local":
        file_list = st.file_uploader(
            "Choose a file...",
            key="meta_data_file_uploader",
            accept_multiple_files=True,
        )
        #st.write("hello")
        #print("hello")
        files_type = st.selectbox("File type", ("", "excel"))
        functionality = st.selectbox("Functionality", ("Generate", ""))
        submit = st.button("Submit")
        if submit:
            print(file_list)
            if functionality == "Generate":
                if file_list:
                    with st.spinner(text="processing..."):
                        (
                            df,
                            all_dataset_full_profile_report,
                            all_dataset_subset_profile_report,
                            summary_list,
                            alert_list,
                            file_name_list,
                            table_column_list,
                        ) = profile_datasets(
                            file_list, source, files_type, functionality
                        )
                        entity_details_formatted = []
                        file_name_list = []
                        for file in file_list:
                            df = pd.read_excel(file, sheet_name=0)
                            data_dict = pd.read_excel(
                                file,
                                sheet_name=1,
                            )
                            match_dict = {}
                            col_names = data_dict["Column"].to_list()
                            descriptions = data_dict["Description"].to_list()
                            for i in range(len(col_names)):
                                match_dict = create_matching_dict(
                                    match_dict, col_names[i], descriptions[i]
                                )
                            file_name_list.append(file.name.replace(".xlsx", ""))
                            formatted_dict = {}
                            formatted_dict["entity_name"] = file.name.replace(
                                ".xlsx", ""
                            )
                            formatted_dict["description"] = ""
                            formatted_dict["attributes"] = []
                            for i in range(len(all_dataset_subset_profile_report)):
                                for j in range(
                                    len(all_dataset_subset_profile_report[i])
                                ):
                                    all_dataset_subset_profile_report[i][j][
                                        "description"
                                    ] = match_dict[
                                        all_dataset_subset_profile_report[i][j]["name"]
                                    ]
                                    formatted_dict["attributes"].append(
                                        all_dataset_subset_profile_report[i][j]
                                    )
                            formatted_dict["summary"] = summary_list[i]
                            formatted_dict["Insights"] = alert_list[i]
                            entity_details_formatted.append(formatted_dict)
                        print(file_name_list)
                        st.session_state["all_dataset_full_profile_report"] = (
                            all_dataset_full_profile_report
                        )
                        st.session_state["entity_details_formatted"] = (
                            entity_details_formatted
                        )
                        st.session_state["file_list"] = file_list
                        st.session_state["df_list"] = None
                        st.session_state["matched_res"] = None
                        st.session_state["file_name_list"] = file_name_list
                        st.session_state["attr_list"] = None
                else:
                    st.error("Please select atleast one file.")

    er_place_holder = st.empty()
    img_pos = st.empty()
    try:
        table_names = ["None"] + st.session_state["file_name_list"]
        expander = st.expander("ENTITIES")
        metadata_curator_search = expander.selectbox(
            "Select the table name", table_names
        )
        schema_container = st.container()

        if metadata_curator_search != "None":

            pos = table_names.index(metadata_curator_search) - 1
            print(pos)
            entity_details = st.session_state["entity_details_formatted"][pos]
            full_profile_rep = st.session_state["all_dataset_full_profile_report"][pos]
            st.session_state["file_name"] = st.session_state["file_name_list"][pos]

            with schema_container:
                st.header(entity_details["entity_name"].upper())
                summary_details = entity_details["summary"]
                insights = entity_details["Insights"]
                cols_df = pd.DataFrame(entity_details["attributes"])
                distinct_cols_lst = list(
                    cols_df[cols_df["distinct_percentage"] == 100]["name"].values
                )

                distinct_cols_str = ", ".join(distinct_cols_lst)
                st.markdown(entity_details["description"])
                st.markdown("<h5>Dataset Highlights</h5>", unsafe_allow_html=True)
                if distinct_cols_lst:
                    st.write(
                        f"- **Potential** **primary** **key:** {distinct_cols_str}"
                    )
                else:
                    prompt = f"""You are sql developer, You will be given the column information of a particular entity.
                                Your job is to predict the potential primary key if it EXISTS based on the column information, else return "There are no potential primary keys".
                                Table name : {entity_details['entity']}
                                Column information: {entity_details['attributes']}

                                Potential primary key:

                                [Primary keys can be composite keys as well with combination of multiple columns]
                                """
                    st.write(
                        "- **Potential** **primary** **key:** "
                        + st.session_state["llm"](prompt)
                    )
                st.write(f"- **Number** **of** **records:** {summary_details['n']}")
                st.write(f"- **Number** **of** **columns:** {summary_details['n_var']}")
                insights = "\n".join(insights).replace("[", "").replace("]", "")
                summarize_insights_prompt = f"""I have a Customer entity Data file which has below insights generated as part of pandas data profiling:
{insights}
Please summarize the insights and produce brief text insights about the  given entity data. Give output in nnumber bullets"""
                insights_gen = st.session_state["llm"](summarize_insights_prompt)

                st.write(f"- **Insights** :\n\n {insights_gen}")
                file_df = pd.read_excel(file_list[pos], sheet_name=0)
                cols_df["attribute_name"] = list(file_df.columns)
                st.table(cols_df)
                ingest_document = st.button(
                    "Add to Knowledge", key=entity_details["entity_name"]
                )

                if ingest_document:
                    #st.write("entered")
                    file_name = st.session_state["file_name"].replace(".csv", "")
                    #st.write(file_name)
                    #st.write(full_profile_rep)
                    vector = restructure_profile_report_for_analysing(
                        [full_profile_rep]
                    )
                    #st.write(vector)

                    entity_dict = {}

                    entity_dict["Table Name"] = file_name
                    #st.write(entity_dict)

                    entity_dict["Table Description"] = entity_details["description"]

                    entity_dict["Insights"] = str(entity_details["Insights"])
                    #st.write(entity_dict)
                    entity_dict["Attribute Name"] = list(file_df.columns)
                    entity_dict["Attribute Description"] = [
                        attribute["description"]
                        for attribute in entity_details["attributes"]
                    ]
                    print("before",entity_dict)
                    entity_dict["Attribute Profile Report"] = vector[0]
                    print("after",entity_dict)
                    entity_df = pd.DataFrame(entity_dict)
                    print("hai")
                    #st.write("entity df",entity_df)

                    engine = create_engine(
                        "mysql+mysqlconnector://root:%s@localhost/data_resolution1"
                        % quote_plus("Root@123")
                    )
                    # entity_df.to_sql(
                    #     "data_unstandardized",
                    #     con=engine,
                    #     if_exists="append",
                    #     index=False,
                    # )
                    
                    st.success("Data has been ingested successfully")
        if len(file_list) > 1:
            er_place_holder.subheader("Entity Relationship Diagram")
            img_pos.markdown(
                "<center> " + st.session_state["er_image"]._repr_html_() + "</center> ",
                unsafe_allow_html=True,
            )
    except Exception as e:
        
        print(e)

if "standardization_rule" not in st.session_state.keys():
    standardization_rule = [
        "match_standardized_name",
        "create_standardized_name",
        "replace_classword_with_data_domain",
        "append_data_domain_name",
        "pascal_case_check",
    ]
    st.session_state["standardization_rule"] = standardization_rule
else:
    standardization_rule = st.session_state["standardization_rule"]


def checkbox_container(data):
    cols = st.columns(9)
    if cols[0].button("Select All"):
        for i in data:
            st.session_state["dynamic_checkbox_" + i] = True
        st.experimental_rerun()
    if cols[1].button("UnSelect All"):
        for i in data:
            st.session_state["dynamic_checkbox_" + i] = False
        st.experimental_rerun()
    for i in data:
        st.checkbox(i, key="dynamic_checkbox_" + i)


def get_selected_checkboxes():
    return [
        i.replace("dynamic_checkbox_", "")
        for i in st.session_state.keys()
        if i.startswith("dynamic_checkbox_") and st.session_state[i]
    ]


def initiate_flow(input_path, rule_list):
    asst_data_model = standardize_names()
    print(input_path)
    sttm_df = pd.read_excel(input_path, sheet_name=0)
    data_dictionary_df = pd.read_excel(input_path, sheet_name=1)
    standardized_names_df = pd.read_excel(input_path, sheet_name=2)
    # st.write("standardized_names_df:",standardized_names_df)
    data_domain_df = pd.read_excel(input_path, sheet_name=3)
    abbr_df = pd.read_excel(input_path, sheet_name=4)
    source_col_names = sttm_df["Column Name"].to_list()
    # st.write("soure_col_names:",source_col_names)
    target_col_names = sttm_df["Column Name.1"].to_list()
    # st.write("target_col_names:",target_col_names)
    standardized_names_list = []

    for i, col_name in enumerate(source_col_names):
        print(
            f"**********************************************\nAttribute name: {target_col_names[i]}"
        )
        name_gen_flag = 0
        if "match_standardized_name" in rule_list:
            standardized_name = asst_data_model.match_standardized_name(
                standardized_names_df, target_col_names[i]
            )
            if standardized_name != "":
                standardized_names_list.append(standardized_name)
                target_col = standardized_name
                name_gen_flag = 1
            data_dict_match = asst_data_model.match_data_dictionary(
                data_dictionary_df, target_col_names[i]
            )

        if name_gen_flag == 0 and "create_standardized_name" in rule_list:
            data_dict_match = asst_data_model.match_data_dictionary(
                data_dictionary_df, target_col_names[i]
            )

            print(f"Match dictionary : {data_dict_match}")
            # st.write("column for standardized name:",source_col_names[i],col_name)
            matched_data_domain, classword = asst_data_model.match_data_domain(
                data_domain_df,
                source_col_names[i],
                target_col_names[i],
                data_dict_match,
            )
            print(f"Match data domain : {matched_data_domain}")
            standardized_name = asst_data_model.create_standardized_name(
                target_col_names[i],
                matched_data_domain,
                data_dict_match,
            )
            target_col = standardized_name
            print(f"Standardized name : {standardized_name}")
        if name_gen_flag == 0 and "replace_classword_with_data_domain" in rule_list:
            target_col = asst_data_model.replace_classword_with_data_domain(
                standardized_name, matched_data_domain, classword
            )
            print(f"Name with classword replace : {target_col}")
        if name_gen_flag == 0 and "append_data_domain_name" in rule_list:
            target_col = asst_data_model.append_data_domain_name(
                target_col, matched_data_domain
            )
            print(f"Name after appending domain : {target_col}")
        if name_gen_flag == 0 and "pascal_case_check" in rule_list:
            target_col = asst_data_model.pascal_case_check(target_col)
            print(f"Final name: {target_col}")
        if name_gen_flag == 0 and len(target_col) > 75:
            abbr_desc_list = abbr_df["Description"].to_list()
            abbr_list = abbr_df["Abbreviation"].to_list()
            for i, abbr_desc in enumerate(abbr_desc_list):
                target_col = target_col.replace(abbr_desc, abbr_list[i])
                if len(target_col) < 75:
                    break
        if name_gen_flag == 0:
            standardized_names_list.append(target_col)

    sttm_df["standardized_name"] = standardized_names_list
    #
    # st.write("standardized",standardized_names_list)
    # st.write("fnction compleeted till standardized")
    # st.write("description name list")
    # sttm_df["Description"]=description_name_list
    # st.write("description Name list:",description_name_list)
    return sttm_df


if datamodel_subtab == "Standardize Data Model":
    st.subheader("Upload Files for Standardization")
    input_files = st.file_uploader("Choose your files...", accept_multiple_files=True)
    rule_application_list = []
    st.subheader("Select rules for standardization")
    checkbox_container(standardization_rule)
    gen_standardized_data_button = st.button(
        "Generate", key="gen_standardized_data_button"
    )
    if gen_standardized_data_button:
        rule_list = get_selected_checkboxes()
        op = {}
        input_name_list = []
        for input_file in input_files:
            sttm_df = initiate_flow(input_file, rule_list)
            op[input_file.name] = sttm_df
            input_name_list.append(input_file.name)
        st.session_state["input_name_list"] = input_name_list
        st.session_state["op"] = op
    try:
        selected_table = st.selectbox(
            "Select the enitity", st.session_state["input_name_list"]
        )
        op = st.session_state["op"]
        try:
            print(op[selected_table].columns)
            temp_df = op[selected_table][
                [
                    "Target Table/Dataset Name",
                    "Column Name",
                    "Column Name.1",
                    "standardized_name",
                ]
            ]
            temp_df.columns = [
                "Table Name",
                "Source attribute name",
                "Old Target attribute Name",
                "Standardized name",
            ]
        except:
            temp_df = op[selected_table]
        edited_df = st.data_editor(temp_df)
        op[selected_table] = edited_df
        add_to_knowledge_button = st.button("Add to knowledge")
        if add_to_knowledge_button:
            
            query = f'Select * from data_unstandardized where `Table Name`="{op[selected_table]["Table Name"].to_list()[0]}";'
            mycursor.execute(query)
            unstandardized_df = DataFrame(
                mycursor.fetchall(),
                columns=[
                    "Table Name",
                    "Table Description",
                    "Insights",
                    "Attribute Name",
                    "Attribute Description",
                    "Attribute Profile Report",
                ],
            )
            #st.write(unstandardized_df)
            match_dict = {}
            old_names = op[selected_table]["Old Target attribute Name"].to_list()
            standardized_names = op[selected_table]["Standardized name"].to_list()
            for i in range(len(old_names)):
                match_dict = create_matching_dict(
                    match_dict, old_names[i], standardized_names[i]
                )

            standardized_name_for_db = []
            old_names_from_db = unstandardized_df["Attribute Name"].to_list()
            profile_reports = unstandardized_df["Attribute Profile Report"].to_list()
            for i, old_name in enumerate(old_names_from_db):
                standardized_name_for_db.append(match_dict[old_name])
                profile_reports[i] = profile_reports[i].replace(
                    old_name, match_dict[old_name]
                )
            unstandardized_df["Attribute Profile Report"] = profile_reports
            unstandardized_df["Attribute Name"] = standardized_name_for_db
            engine = create_engine(
                "mysql+mysqlconnector://root:%s@localhost/data_resolution1"
                % quote_plus("Root@123")
            )
            #st.write("unstandardized_df",unstandardized_df)
            # unstandardized_df.to_sql(
            #     "data_standardized", con=engine, if_exists="append", index=False
            # )
            st.success("Data has been ingested successfully")
            processed_entity_df = pd.read_csv(
                r"output/processed_entities.csv"
            )
            table_name_list = list(unstandardized_df["Table Name"].unique())
            for table_name in table_name_list:
                if (
                    len(
                        processed_entity_df[
                            processed_entity_df["processed_entity_names"] == table_name
                        ]
                    )
                    == 0
                ):
                    enitites_to_process = pd.read_csv(
                        r"output/deduplication.csv"
                    )
                    temp_df = pd.DataFrame(
                        [table_name], columns=["entities_to_process"]
                    )

                    pd.concat([enitites_to_process, temp_df]).reset_index(
                        drop=True
                    ).to_csv(
                        r"output/deduplication.csv",
                        index=False,
                    )
                    processed_entities = pd.read_csv(
                        r"output/processed_entities.csv"
                    )
                    temp_df = pd.DataFrame(
                        [table_name], columns=["processed_entity_names"]
                    )
                    pd.concat([processed_entities, temp_df]).reset_index(
                        drop=True
                    ).to_csv(
                        r"output/processed_entities.csv",
                        index=False,
                    )

    except Exception as e:
        print(e)






if datamodel_subtab=="Deduplication":
    llm=st.session_state["llm"]
    #st.session_state["product image"] =""
    

    entities_to_be_processed=pd.read_csv("output/deduplication.csv")
    #select the standardized data to be processed
    mydb = mysql.connector.connect(
    host=config["MYSQL"]["host"], user=config["MYSQL"]["user"], password=config["MYSQL"]["password"], database="data_resolution1")
    mycursor=mydb.cursor()
    mycursor.execute("select * from data_deduplication")
    deduplicated_entities=pd.DataFrame(mycursor.fetchall(), columns=[desc[0] for desc in mycursor.description])
    condition_met = False

    # Iterate over entities
    for entities in entities_to_be_processed["entities_to_process"]:
        if deduplicated_entities.empty or entities not in deduplicated_entities["Table Name"].unique():
            # Set flag to True if the condition is met
            condition_met = True

    # Check if the condition was met in any iteration
    if condition_met:
        # Ask for domain details and input file details only once
        domain_file = st.file_uploader("Select the entity Domain Level Info file...", type="xlsx", key="domain_file")
        if domain_file is not None:
            domain_config = pd.read_excel(domain_file)
            
        input_file = st.file_uploader("Select the Hierarchical Config file...", type="json", key="hierarchical_file")
        if input_file is not None:
            json_data = json.loads(input_file.read())
    if condition_met==True:
        if st.button("Analyze",key="Duplicate Submit"):
            for entities in entities_to_be_processed["entities_to_process"]:
                if deduplicated_entities.empty or entities not in deduplicated_entities["Table Name"].unique():
                    with st.spinner("Processing....."):
                        mycursor.execute(f"SELECT * FROM data_standardized WHERE `Table Name`='{entities}'")
                        entities_added_to_knowledge = pd.DataFrame(mycursor.fetchall(), columns=[desc[0] for desc in mycursor.description])

                        
                        grouped_entities={}
                        grouped_entities["Table Name"]=entities
                        #st.write(grouped_entities)
                        
                        grouped_entities["Attribute Name"]=entities_added_to_knowledge["Attribute Name"].tolist()
                        
                        grouped_entities["Attribute Description"]=entities_added_to_knowledge["Attribute Description"].tolist()
                        grouped_entities["Attribute Profile Report"]=entities_added_to_knowledge["Attribute Profile Report"].tolist()
                        grouped_entities["Insights"]=entities_added_to_knowledge["Insights"]
                        #df=pd.DataFrame(grouped_entities)
                        
                        attribute_details = {name: description for name, description in zip(grouped_entities["Attribute Name"], grouped_entities["Attribute Description"])}
                        table_type_result = get_table_type(entities, attribute_details, llm)
                        domain_result=get_domain_subject_area(entities, attribute_details, llm, domain_config)
                        table_type_lines = table_type_result.strip().split("\n")
                        for idx, line in enumerate(table_type_lines):
                            if idx == 1:
                                grouped_entities["Table_Type"] = line.split(":", 1)[1].strip()
                        
                        lines = domain_result.strip().split("\n")
                        print("lines", lines)
                        for idx, line in enumerate(lines):  # Iterate over lines with index
                            if idx == 1:
                                grouped_entities["Domain"] = line.split(":", 1)[1].strip()
                            elif idx == 2:
                                grouped_entities["Subject_Area"] = line.split(":", 1)[1].strip()
                            elif idx == 3:
                                grouped_entities["Subject_Area_Subset"] = line.split(":", 1)[1].strip()
                        #Convert to dataframe
                        attribute_types = []
                        for attribute, attribute_description, attribute_profile_report in zip(grouped_entities["Attribute Name"], grouped_entities["Attribute Description"], grouped_entities["Attribute Profile Report"]):
                            table_type_result = get_attribute_type(entities, attribute, attribute_description, attribute_profile_report, llm)
                            attribute_types.append(table_type_result)

                        grouped_entities["Attribute Type"] = attribute_types
                        #st.write(grouped_entities)
                        #st.write(type(grouped_entities))
                        # Extracting attribute list based on attribute type
                        attribute_list = [attr_name for attr_name, attr_type in zip(grouped_entities.get("Attribute Name", []), grouped_entities.get("Attribute Type", [])) if attr_type in ["Dimension", "Key"]]
                        #st.write("attribute list",attribute_list)
                        domain = grouped_entities["Domain"]
                        subject_area = grouped_entities["Subject_Area"]
                        subject_area_subset = grouped_entities["Subject_Area_Subset"]
                        table_type = grouped_entities["Table_Type"]
                        grouped_entities["Granularity Level"]=str(get_granularity_data(entities,domain,subject_area,subject_area_subset,table_type,attribute_list,json_data,llm))
                        purpose_type = []
                        calculation_type = []
                        for attr_name, attr_description, attr_profile in zip(grouped_entities.get("Attribute Name", []), grouped_entities.get("Attribute Description", []), grouped_entities.get("Attribute Profile Report", [])):
                            calculation, purpose = get_purpose_of_calculation(domain, subject_area, attr_name, attr_description, attr_profile, llm)
                            calculation_type.append(calculation)
                            purpose_type.append(purpose)

                        grouped_entities["Possible Method Of Calculation"] = calculation_type
                        grouped_entities["Purpose Of Measure"] = purpose_type
                        data_deduplication= pd.DataFrame(grouped_entities)
                        #st.write(data_deduplication)
                        #insert_data_into_pinecone(data_deduplication)
                        engine = create_engine(
                                "mysql+mysqlconnector://root:%s@localhost/data_resolution1"
                                % quote_plus("Root@123")
                        )
                        # data_deduplication.to_sql(
                        #         "data_deduplication",
                        #         con=engine,
                        #         if_exists="append",
                        #         index=False,
                        #     )
                        st.success("Data has been ingested successfully")
                    #connection.commit()
                        entities_to_be_processed= entities_to_be_processed[entities_to_be_processed['entities_to_process'] != entities]
                        entities_to_be_processed.to_csv('output/deduplication.csv', index=False)
                else:
                    st.write(f"{entities} already added to knowledge")
                    entities_to_be_processed= entities_to_be_processed[entities_to_be_processed['entities_to_process'] != entities]
                    entities_to_be_processed.to_csv('output/deduplication.csv', index=False)

    condition_met=False
    with open("config/hierarchy_config_1.json", "r") as file:
        hierarchy_config= json.load(file)
    col1, col2 = st.columns(2)
    value=''
    with col1:
        col3,col4= st.columns(2)
        with col3:
            if 'Product Hierarchy' not in st.session_state:
                st.session_state["Product Hierarchy"]=False
            if st.button("Product Hierarchy",key="Product") or st.session_state["Product Hierarchy"]:
                value="Product Hierarchy"
                img=generate_mermaid_code(hierarchy_config,value)
                st.session_state["product image"]=img
                st.write(st.session_state["product image"])
                st.session_state["Product Hierarchy"]=True

        with col4:
            if 'Customer_Hierarchy' not in st.session_state:
                st.session_state["Customer_Hierarchy"]=False
            if st.button("Customer Hierarchy",key="Customer") or st.session_state["Customer_Hierarchy"]:
                value="Customer_Hierarchy"
                img=generate_mermaid_code(hierarchy_config,value)
                st.session_state["customer image"]=img
                st.write(st.session_state["customer image"])
                st.session_state["Customer_Hierarchy"]=True

    
    mydb = mysql.connector.connect(
    host=config["MYSQL"]["host"], user=config["MYSQL"]["user"], password=config["MYSQL"]["password"], database="data_resolution1")
    mycursor=mydb.cursor()
    mycursor.execute("select * from data_deduplication")
    deduplicated_entities=pd.DataFrame(mycursor.fetchall(), columns=[desc[0] for desc in mycursor.description])
    domain=st.multiselect("Select the Domain",deduplicated_entities["Domain"].unique(),key="domain related")
    for each_domain in domain:
            subject_area=st.multiselect(f"Select the Subject Area for {each_domain}",deduplicated_entities[deduplicated_entities["Domain"]==each_domain]["Subject_Area"].unique())
            if st.button("Find Duplicates",key="Analyse"):
                for each_subject in subject_area:
                    get_duplicates(each_domain,each_subject,deduplicated_entities,llm)