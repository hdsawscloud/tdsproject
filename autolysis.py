#!/bin/env uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib",
#     "openai",
#     "pandas",
#     "python-dotenv",
#     "seaborn",
# ]
# ///

# configure env variables
from dotenv import load_dotenv
load_dotenv()

import os
from openai import OpenAI
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if "AIPROXY_TOKEN" not in os.environ:
    print("could not find AIPROXY_TOKEN in env")
    exit(1)

client = OpenAI(
    base_url="https://aiproxy.sanand.workers.dev/openai/",
    api_key=os.environ["AIPROXY_TOKEN"],
)

# there are three types of columns
# numeric: numbers
# text enum: unique values < len(dataset)/3
# free text: everything else
#
# they will be analyzed as
# numeric:
#   - bar graph of mean and std deviation for all columns
#   - correlation matrix
#   - line graph, x/y axis to be determined by llm
# text enum:
#   - frequency analysis histogram
#   - top 10 records analysis via llm
# free text:
#   - word cloud
#   - keyword histogram
#   - top 10 keywords analysis via llm

class Dataset:
    df: pd.DataFrame
    numeric: pd.DataFrame
    enum: pd.DataFrame
    text: pd.DataFrame

def load_dataset(path: str) -> Dataset:
    dataset = Dataset()
    dataset.df = pd.read_csv(path)
    dataset.numeric = dataset.df.select_dtypes(include="number")
    
    text_columns = dataset.df.select_dtypes(exclude="number").astype(str)
    # create a filter to select columns that have count of unique values < count of values/3
    enum_filter = text_columns.nunique() < text_columns.count()/3
    dataset.enum = text_columns[enum_filter]
    dataset.text = text_columns[~enum_filter]

def prompt_llm(system_prompt: str, user_prompt: str) -> str:
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": user_prompt},
        ],
    )
    return completion.choices[0].message.content

def analyze_file(dataset: Dataset) -> list[str]:
    output: list[str] = []
    
    # generate title based on contents
    title = ""
    
    output += [
        f"# {title}",
        ""
    ]
    
    numeric_analysis += analyze_numeric(dataset, output)
    # enum_analysis = analyze_enum(dataset, output)
    # text_analysis = analyze_text(dataset, output)
    
    # generate analysis summary
    
    return output

def analyze_numeric(dataset: Dataset, output: list[str]) -> list[str]:
    analysis = []
    output += [
        "## Anaylsis of Numeric Columns",
        ""
    ]
    
    # 1. bar graph of mean and std deviation for all numeric columns
    output += [ 
        "### Bar Graph of Mean and Std. Deviation"
        "![bar graph of mean and std deviation for all numeric columns](./numeric-bar-graph.png)",
        ""
    ]
    
    # 2. correlation matrix
    output += [
        "### Correlation Matrix",
    ]
    if dataset.numeric.columns < 2:
        output += [
            "unable to plot: dataset needs at least 2 numeric columns to plot a correlation matrix",
            ""
        ]
    else:
        output += [
            "![correlation matrix](./numeric-correlation-matrix.png)",
            ""
        ]
    
    # 3. line graph of interesting column pairs
    output += [
        "### Line Graph of Interesting Column Pairs",
    ]
    if dataset.df.columns < 2:
        output += [
            "unable to plot: dataset needs at least 2 columns to plot a line graph",
            ""
        ]
    else:
        # prompt llm for interesting column pairs
        # generate line graph based on pairs
        output += [
            "![line graph of interesting column pairs](./numeric-line-graph.png)",
            ""
        ]

def analyze_enum(enum: pd.DataFrame) -> list[str]:
    pass
    # text enum:
    #   - frequency analysis histogram
    #   - top 10 records analysis via llm

def analyze_text(enum: pd.DataFrame) -> list[str]:
    pass
    # free text:
    #   - word cloud
    #   - keyword histogram
    #   - top 10 keywords analysis via llm


def main():
    filename = sys.argv[1]
    if not os.path.isfile(filename):
        print(f"file {filename} is not a file")
        print(f"usage: ./autolysis.py filename.csv")
        exit(1)
    
    dataset = load_dataset(filename)
    print(f"loaded dataset from {filename}")
    
    print("starting analysis")
    analysis = analyze_file(dataset)
    with open("README.md", "w") as f:
        f.write("\n".join(output))
    print("analysis complete, written to README.md")

if __name__ == "__main__":
    main()
