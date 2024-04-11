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
import json


def mm(graph):
    graphbytes = graph.encode("utf8")
    base64_bytes = base64.b64encode(graphbytes)
    base64_string = base64_bytes.decode("ascii")
    output = Image(url="https://mermaid.ink/img/" + base64_string)
    return output


def get_matching_attribute(metadata, llm, embeddings):
    index_name = "llm-accelerator-openai"
    pinecone.init(
        api_key="6bc62698-733d-481f-af2d-b1cb8bd61312", environment="us-west4-gcp-free"
    )
    database_metadata = Pinecone.from_existing_index(
        index_name, embeddings, namespace="metadata_repository"
    )
    attribute_search_prompt_template = """You are data analysis expert and you are trying to identify the correct attribute based on the sample data and data profile for an attribute given the following context.
Context: Information stored as vectors are pandas data profile report for different database attributes. Each document corresponds to a database attribute that holds profile reports (e.g. p_distinct, p_unique, type, p_missing, max_length, data_pattern, max_length, min_length, mean_length, median_length  ..etc.). Go through the properties of the the input data profile that does not contain database attribute_name, try to match different properties (which are available) of the input data profile with the existing profiles available as vectors then give the best possible match attribute_name list as output. If there are multiple matches please give all of them as output with explanation.
Hint:
    1) Try to match the properties (e.g. p_missing, p_unique, p_distinct ..etc.) in the input data profile with the data profile property available as vector
    2) The profile property data_pattern contain list of sample values. Try to understand the data pattern (if available) in the input data profile by going through each element in the data_pattern via regular expression technique. The data pattern wherever available in the vectors are same way contain list of sample values of database attributes. Try to match the data_pattern in the input profile with vectors.

Input Data Profile: {question}

============
{summaries}
============

Answer: 
---BEGIN FORMAT TEMPLATE---
${{attribute_name}}: [explanation]
${{attribute_name}}: [explanation]
---END FORMAT TEMPLATE---
"""
    attrib_match_chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=database_metadata.as_retriever(),
        verbose=True,
        chain_type_kwargs={
            "prompt": PromptTemplate(
                template=attribute_search_prompt_template,
                input_variables=["summaries", "question"],
            ),
        },
    )
    res_list = []
    attr_list = []
    for table_metadata in metadata:
        res = []
        attr = []
        for rec in table_metadata:
            attrib_match_chain_response = attrib_match_chain(
                {"question": rec, "summaries": ""}, return_only_outputs=True
            )
            op_str = attrib_match_chain_response["answer"]
            res.append(op_str)
            op_str = op_str.replace("\n", " ")
            indexes = [i for i, ltr in enumerate(op_str) if ltr == ":"]
            extr_attr_list = []
            for ind in indexes:
                flag = False
                name = ""
                for i in range(ind - 1, -1, -1):
                    if i == 0:
                        name = op_str[i : i + 1] + name
                        extr_attr_list.append(name)
                    if flag == True and op_str[i : i + 1] == " ":
                        print(name)
                        extr_attr_list.append(name)
                        break
                    elif flag == False and op_str[i : i + 1] == " ":
                        continue
                    else:
                        flag = True
                        name = op_str[i : i + 1] + name
            extr_attr_list.append("Custom")
            attr.append(extr_attr_list)
        attr_list.append(attr)
        res_list.append(res)
    return res_list, attr_list


def restructure_profile_report_for_analysing(all_dataset_full_profile_report):
    #st.write("profile entered")
    metadata_list = []
    for i in range(len(all_dataset_full_profile_report)):
        temp_dict = all_dataset_full_profile_report[i]["profiling"].copy()
        attrs = temp_dict["variables"].keys()
        metadata = []
        for attr in attrs:
            print(temp_dict["variables"][attr])
            if (
                temp_dict["variables"][attr]["type"] == "Text"
                or temp_dict["variables"][attr]["type"] == "Categorical"
            ):
                print("Text or Ctegorial",attr)
                del temp_dict["variables"][attr]["value_counts_without_nan"]
                del temp_dict["variables"][attr]["value_counts_index_sorted"]
                del temp_dict["variables"][attr]["length_histogram"]
                del temp_dict["variables"][attr]["histogram_length"]
                del temp_dict["variables"][attr]["character_counts"]
                del temp_dict["variables"][attr]["n_characters_distinct"]
                del temp_dict["variables"][attr]["n_characters"]
                del temp_dict["variables"][attr]["category_alias_values"]
                del temp_dict["variables"][attr]["block_alias_values"]
                del temp_dict["variables"][attr]["block_alias_counts"]
                del temp_dict["variables"][attr]["n_block_alias"]
                del temp_dict["variables"][attr]["block_alias_char_counts"]
                del temp_dict["variables"][attr]["script_counts"]
                del temp_dict["variables"][attr]["n_scripts"]
                del temp_dict["variables"][attr]["script_char_counts"]
                del temp_dict["variables"][attr]["category_alias_counts"]
                del temp_dict["variables"][attr]["n_category"]
                del temp_dict["variables"][attr]["category_alias_char_counts"]
                del temp_dict["variables"][attr]["word_counts"]
                del temp_dict["variables"][attr]["n_distinct"]
                del temp_dict["variables"][attr]["n_unique"]
                del temp_dict["variables"][attr]["n_missing"]
                del temp_dict["variables"][attr]["n"]
                del temp_dict["variables"][attr]["count"]
                del temp_dict["variables"][attr]["memory_size"]
                try:
                    temp_dict["variables"][attr]["data_pattern"] = list(
                        temp_dict["variables"][attr]["first_rows"].values()
                    )
                    del temp_dict["variables"][attr]["first_rows"]
                except:
                    pass
                temp_dict["variables"][attr]["entity_name"] = (
                    all_dataset_full_profile_report[i]["dataset_name"]
                )
                temp_dict["variables"][attr]["attribute_name"] = attr
                metadata.append(str(temp_dict["variables"][attr]))
            elif temp_dict["variables"][attr]["type"] == "Numeric":
                print("Numeric",attr)

                del temp_dict["variables"][attr]["histogram"]
                del temp_dict["variables"][attr]["value_counts_without_nan"]
                del temp_dict["variables"][attr]["value_counts_index_sorted"]
                del temp_dict["variables"][attr]["n_distinct"]
                del temp_dict["variables"][attr]["n_unique"]
                del temp_dict["variables"][attr]["n_missing"]
                del temp_dict["variables"][attr]["n"]
                del temp_dict["variables"][attr]["count"]
                del temp_dict["variables"][attr]["memory_size"]
                del temp_dict["variables"][attr]["n_infinite"]
                del temp_dict["variables"][attr]["n_negative"]
                del temp_dict["variables"][attr]["n_zeros"]
                del temp_dict["variables"][attr]["sum"]
                del temp_dict["variables"][attr]["min"]
                del temp_dict["variables"][attr]["range"]
                del temp_dict["variables"][attr]["5%"]
                del temp_dict["variables"][attr]["max"]
                del temp_dict["variables"][attr]["25%"]
                del temp_dict["variables"][attr]["50%"]
                del temp_dict["variables"][attr]["75%"]
                del temp_dict["variables"][attr]["95%"]
                try:
                    temp_dict["variables"][attr]["data_pattern"] = list(
                        temp_dict["variables"][attr]["first_rows"].values()
                    )
                    del temp_dict["variables"][attr]["first_rows"]
                except:
                    pass
                temp_dict["variables"][attr]["entity_name"] = (
                    all_dataset_full_profile_report[i]["dataset_name"]
                )
                temp_dict["variables"][attr]["attribute_name"] = attr
                metadata.append(str(temp_dict["variables"][attr]))
            elif temp_dict["variables"][attr]["type"] == "DateTime":
                print("Datetime",attr)
                del temp_dict["variables"][attr]["histogram"]
                del temp_dict["variables"][attr]["value_counts_without_nan"]
                del temp_dict["variables"][attr]["value_counts_index_sorted"]
                temp_dict["variables"][attr]["entity_name"] = (
                    all_dataset_full_profile_report[i]["dataset_name"]
                )
                temp_dict["variables"][attr]["attribute_name"] = attr
                try:
                    temp_dict["variables"][attr]["data_pattern"] = list(
                        temp_dict["variables"][attr]["first_rows"].values()
                    )
                    del temp_dict["variables"][attr]["first_rows"]
                except:
                    pass
                metadata.append(str(temp_dict["variables"][attr]))
            else:
                print(temp_dict["variables"][attr]['type'],attr)
                try:
                    del temp_dict["variables"][attr]["value_counts_without_nan"]
                except:
                    pass
                try:
                    del temp_dict["variables"][attr]["value_counts_index_sorted"]
                except:
                    pass
                try:
                    del temp_dict["variables"][attr]["n_distinct"]
                except:
                    pass
                try:
                    del temp_dict["variables"][attr]["n_unique"]
                except:
                    del temp_dict["variables"][attr]["n_missing"]
                try:
                    del temp_dict["variables"][attr]["n"]
                except:
                    pass
                try:
                    del temp_dict["variables"][attr]["memory_size"]
                    temp_dict["variables"][attr]["data_pattern"] = list(
                        temp_dict["variables"][attr]["first_rows"].values()
                    )
                    del temp_dict["variables"][attr]["first_rows"]
                except:
                    pass
                temp_dict["variables"][attr]["entity_name"] = (
                    all_dataset_full_profile_report[i]["dataset_name"]
                )
                temp_dict["variables"][attr]["attribute_name"] = attr
        metadata_list.append(metadata)
    #st.write(metadata_list)
    return metadata_list


def profile_datasets(file_list, upload_type, file_type, functionality):
    all_dataset_full_profile_report = []
    all_dataset_subset_profile_report = []
    summary_list = []
    alert_list = []
    file_name_list = []
    df_list = []
    table_column_list = []
    print(file_list)
    ## For each file in the directory get profile report
    for idx in range(len(file_list)):
        # Read the file into a pandas dataframe
        if upload_type == "Browse from Local":
            file_name = file_list[idx].name
        elif upload_type == "GCP GCS":
            file_name = file_list[idx].split("/")[-1]
        print(file_name)
        if file_type == "excel":
            if functionality == "Generate":
                # df = pd.read_csv(file_list[idx])
                df = pd.read_excel(file_list[idx], sheet_name=0)
            else:
                df = pd.read_csv(file_list[idx], header=None)

        # Initialize dict variable to store the profile report for each file
        dataset_profiling_dict = {}

        # Generate profile report
        profile = ProfileReport(df, title="Profiling Report")
        profiling_json = profile.to_json()
        profile_json_obj = json.loads(profiling_json)

        # Store the profile report in the python dict
        dataset_profiling_dict["dataset_name"] = file_name
        dataset_profiling_dict["profiling"] = profile_json_obj

        # Store the profile report for each file in the consolidated dict
        all_dataset_full_profile_report.append(dataset_profiling_dict)

        # Get only the attribute name and type of the attribute
        profile_subset = []
        col_dict = {}
        for key, value in profile_json_obj["variables"].items():
            inner_dict = {}
            col_dict[key] = profile_json_obj["variables"][key]["type"]
            inner_dict["name"] = key
            inner_dict["type"] = profile_json_obj["variables"][key]["type"]
            if inner_dict["type"] != "Unsupported":
                inner_dict["distinct_percentage"] = (
                    profile_json_obj["variables"][key]["p_distinct"] * 100
                )
                inner_dict["missing_percentage"] = (
                    profile_json_obj["variables"][key]["p_missing"] * 100
                )
                if inner_dict["type"] == "Categorical":
                    inner_dict["distinct_count"] = profile_json_obj["variables"][key][
                        "n_distinct"
                    ]
                if inner_dict["type"] == "Text":
                    inner_dict["distinct_count"] = profile_json_obj["variables"][key][
                        "n_distinct"
                    ]
                if inner_dict["type"] == "Numeric":
                    inner_dict["min"] = profile_json_obj["variables"][key]["min"]
                    inner_dict["max"] = profile_json_obj["variables"][key]["max"]

            profile_subset.append(inner_dict)
        file_name_list.append(file_name)
        print(type(df))
        df_list.append(df)
        summary_list.append(profile_json_obj["table"])
        alert_list.append(profile_json_obj["alerts"])
        all_dataset_subset_profile_report.append(profile_subset)
        temp = {}
        temp["table_name"] = file_name
        temp["attributes"] = col_dict
        table_column_list.append(temp)
    print(len(summary_list), len(alert_list), len(all_dataset_subset_profile_report))
    #st.write("df list",df_list)
    #st.write(all_dataset_full_profile_report)
    
    return (
        df_list,
        all_dataset_full_profile_report,
        all_dataset_subset_profile_report,
        summary_list,
        alert_list,
        file_name_list,
        table_column_list,
    )


def get_description_and_mermaid_code(
    all_dataset_subset_profile_report, summary_list, alert_list, llm, table_column_list
):
    metadata_gen_prompt_template_part1 = """You are an expert in data modelling and business domain concepts and you are generating the business meaning or context of database Attribute and Entity.
    Context: you will be provided a list of JSON as an input. Each JSON object is list of attributes of an Entity. 
    Try your best to generate description for each attribute by inferring the business context of the attribute from its name and additional details provided. Analyze the list of attributes in an entity and try best to generate a valid name for the entity (which best explains all the attributes) based on your domain knowledge. The entity name should be compatible for any database following ANSI standard, each entity should have valid description generated based on the attribute details provided. Use underscore(_) as a separator for entity name.
    Generate output in the below format:
    [{"entity": entity_name, "description":"entity_description", "attributes": [{"attribute_name": column_name, "type":column_type, "description": column_description, "missing_percentage": missing_percentage, "distinct_percentage": distinct_percentage, .. other columns}, ....] }, ...]

    """
    metadata_gen_prompt_template_part2 = f"""
    Input: {all_dataset_subset_profile_report}

    Answer:

    """
    response = llm(
        metadata_gen_prompt_template_part1 + metadata_gen_prompt_template_part2
    )
    print(response)
    entity_details_formatted = json.loads(response)
    print(len(summary_list), len(alert_list), len(entity_details_formatted))
    for i in range(len(entity_details_formatted)):
        entity_details_formatted[i]["summary"] = summary_list[i]
        entity_details_formatted[i]["Insights"] = alert_list[i]
    context_str = ""
    mermaid_code = "erDiagram "
    for doc in table_column_list:
        context_str = (
            context_str
            + "\n"
            + "Table name : "
            + doc["table_name"].replace(".csv", "")
            + "\n"
            + "Attributes : "
        )
        mermaid_code = mermaid_code + doc["table_name"].replace(".csv", "") + "{ "
        for col in doc["attributes"].keys():
            context_str = context_str + col + ","
            mermaid_code = mermaid_code + col + f"  {doc['attributes'][col]}  "
        mermaid_code = mermaid_code + "}"
    define_relationship_prompt = f"""
You are a software engineer with immense knowledge in generating ER diagrams. You will be given a list of table information. By going through the table name give me the relationships between the table.
Table information:
{context_str}

These relationships will be later incorporated into a mermaid code. In mermaid code the relationship between entities are defined as follows
|o--o| : Zero or one
||--||	: Exactly one
}}o--o{{ : Zero or more (no upper limit)
}}|--|{{ : One or more (no upper limit)


Relationships:
---BEGIN FORMAT TEMPLATE---
{{table name1}} {{Relationship symbol}} {{table name2}} : {{single_word_describing_relationship}}
{{table name1}} {{Relationship symbol}} {{table name2}} : {{single_word_describing_relationship}}
---END FORMAT TEMPLATE---

"""
    mermaid_code = mermaid_code + llm(define_relationship_prompt)
    formatted_mermaid_code = mermaid_code
    img = mm(formatted_mermaid_code)
    print(formatted_mermaid_code)

    return entity_details_formatted, img
