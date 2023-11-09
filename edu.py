####--- IMPORTS ---####

import pandas as pd 
import numpy as np
import plotly.express as px


#Dataset
df = pd.read_csv('D:/Python/Dataset/Global_Education.csv', encoding='latin-1')
df.head()

## Reading proficiencies
fig = px.choropleth(df, locations='Countries and areas', locationmode='country names',
                    color='Grade_2_3_Proficiency_Reading', range_color=[0, 100],
                    title='Proficiency in Reading by Country')
fig.show()


## Competition Rates

completion_columns = ['Completion_Rate_Primary_Male', 'Completion_Rate_Primary_Female',
                      'Completion_Rate_Lower_Secondary_Male', 'Completion_Rate_Lower_Secondary_Female',
                      'Completion_Rate_Upper_Secondary_Male', 'Completion_Rate_Upper_Secondary_Female']
fig = px.line(df, x='Countries and areas', y=completion_columns,
              title='Completion Rates Over Different Education Levels')
fig.update_layout(xaxis_tickangle=-45)
fig.show()


## Reading and mathematics

fig = px.scatter(df, x="Grade_2_3_Proficiency_Reading", y="Grade_2_3_Proficiency_Math", text="Countries and areas",
                 title="Proficiency in Reading vs. Math for Grade 2-3 Students",
                 labels={"Grade_2_3_Proficiency_Reading": "Reading Proficiency", "Grade_2_3_Proficiency_Math": "Math Proficiency"},
                 color="Youth_15_24_Literacy_Rate_Male", size="Youth_15_24_Literacy_Rate_Female")

fig.update_traces(textposition='top center', marker=dict(size=12, opacity=0.7))
fig.show()


## Pre-Primary Age

fig = px.bar(df, x='Countries and areas', y=['OOSR_Pre0Primary_Age_Male', 'OOSR_Pre0Primary_Age_Female'],
             title='Out-of-School Rates by Gender (Pre-Primary Age)')
fig.update_layout(xaxis_tickangle=-45)
fig.show()


### Unemployment

fig = px.scatter(df, x='Countries and areas', y='Unemployment_Rate', color='Unemployment_Rate',
                 title='Unemployment Rates Across Countries')
fig.update_layout(xaxis_tickangle=-45)
fig.show()


### Unemployment vs proficiency

fig = px.scatter_3d(df, x='Unemployment_Rate', y='Grade_2_3_Proficiency_Reading', z='Grade_2_3_Proficiency_Math',
                   color='Countries and areas', title='Unemployment vs. Proficiency')
fig.show()