import dash
from dash import dcc, html, Input, Output, State
import dash_cytoscape as cyto
import pandas as pd
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF
import networkx as nx

ex = Namespace("http://www.example.org/RealEstateAnalytics#")

xl_file = "Real_Estate_Analytics.xlsx"

df = pd.read_excel(xl_file)

g = Graph()

for index, row in df.iterrows():
    article = str(row['Article'])
    
    project_uri = ex[article.replace(" ", "_")]  
    g.add((project_uri, RDF.type, ex.AnalyticsProject))
    
    for column in df.columns:
        if column != 'Article':  
            value = row[column]
            g.add((project_uri, ex[column.replace(" ", "")], Literal(value)))

for s, p, o in list(g):
    if isinstance(o, Literal) and (pd.isna(o.value) or o.value == '' or o.value==' '):
        g.remove((s, p, o))

rdf_text = g.serialize(format='turtle')

output_file = "rdf_triples.txt"  
with open(output_file, 'w', encoding='utf-8') as file:
    file.write(rdf_text)

nx_graph = nx.DiGraph()  

for s, p, o in g:
    nx_graph.add_node(s)
    nx_graph.add_node(o)
    nx_graph.add_edge(s, o, label=p.split('#')[-1])

app = dash.Dash(__name__, suppress_callback_exceptions=True) 

initial_seed = 42
initial_k = 3

pos = nx.spring_layout(nx_graph, seed=initial_seed, k=initial_k)

cyto_elements = []

for node in nx_graph.nodes():
    x, y = pos[node]
    cyto_elements.append({'data': {'id': node, 'label': node.split('#')[-1]}})

for edge in nx_graph.edges():
    source, target = edge
    cyto_elements.append({'data': {'source': source, 'target': target, 'label': nx_graph.edges[edge]['label']}})


cyto_stylesheet = [
    {'selector': 'node', 'style': {'label': 'data(label)', 'font-size': '20px'}},
    {'selector': 'edge', 'style': {'label': 'data(label)', 'font-size': '8px', 'curve-style': 'bezier', 'target-arrow-shape': 'triangle', 'arrow-scale': 2}},  
    {'selector': 'node[project]', 'style': {'background-color': 'blue', 'shape': 'square'}},  
    {'selector': 'node', 'style': {'content': 'data(label)'}},
    {'selector': 'edge', 'style': {'content': 'data(label)'}},
    {'selector': ':parent', 'style': {'background-opacity': 0.333}},
    {'selector': '.highlighted', 'style': {'background-color': 'yellow', 'line-color': 'yellow', 'target-arrow-color': 'yellow', 'source-arrow-color': 'yellow'}}
]


predicate_color_map = {
    'utilizesAlgorithmForAnalysis': '#FF5733',  
    'carriesOutAnalyticsOfType': '#33FF57',  
    'utilizesModelForAnalysis': '#5733FF',  
    'extractsDataUsing': '#FF33EC', 
    'utilizesDataOfType': '#33ECFF', 
    'carriesOutAnalyticsOfType': '#FF3366',  
    'Reference': '#000',  
    'resultsInInteractiveResult': '#3366FF', 
    'verifiesResultsUsing': '#FFCC33',  
    'utilizesSoftwareForAnalysis': '#33FFCC',  
    'utilizesDataSource': '#CC33FF',  
    'definesDataset': '#FF3333',  
    'utilizesAnalysisMethod': '#33FF33',  
    'resultsInReport': '#3333FF', 
    'resultsInGraphicalResult': '#FF99FF',  
    'integratesDataThrough': '#99FF99',  
    'concernsSystemOrObjectOrMeasureOfInterest': '#9999FF',  
    'resultsInModel': '#FFFF33',  
    'utilizesProjectDocumentation': '#33FFFF'  
}

for predicate, color in predicate_color_map.items():
    cyto_stylesheet.append({
        'selector': f'edge[label="{predicate}"]',
        'style': {'line-color': color, 'target-arrow-color': color}
    })


cyto_stylesheet.append({
    'selector': 'node[id^="http://www.example.org/RealEstateAnalytics"]',
    'style': {
        'background-color': 'blue',
        'shape': 'square'
    }
})

subjects = set()
objects = set()
predicates = set()
for s,p,o in g:
    subjects.add(s)
    objects.add(o)

layout_algorithms = [
    {'label': 'circle', 'value': 'circle'},
    {'label': 'grid', 'value': 'grid'},
    {'label': 'cose', 'value': 'cose'},
    {'label': 'random', 'value': 'random'},  
    {'label': 'concentric', 'value': 'concentric'},  
    {'label': 'breadthfirst', 'value': 'breadthfirst'},

]

app.layout = html.Div([
    html.H1("Real Estate Analytics Graph"),
    html.Div([
        html.Label("Filter by Subjects:"),
        dcc.Dropdown(
            id='subject-dropdown',
            options=[{'label': s.split('#')[-1], 'value': s} for s in subjects],
            multi=True
        ),
        html.Label("Filter by Predicates:"),
        dcc.Dropdown(
            id='predicate-dropdown',
            options=[{'label': p, 'value': p} for p in predicate_color_map.keys()],
            multi=True
        ),
        html.Label("Filter by Objects:"),
        dcc.Dropdown(
            id='object-dropdown',
            options =[{'label': s.split('#')[-1], 'value': o} for o in objects],
            multi=True
        ),
        html.Button('Update Graph', id='Update_Graph-Button', n_clicks=0),
        html.Label("Select Layout Algorithm:"),
        dcc.Dropdown(
            id='layout-dropdown',
            options=layout_algorithms,
            value='circle',  
            clearable=False
        )
    ]),
    html.Div(id='cytoscape-container'),
    html.Div(id='legend-container', style={'margin-top': '20px'}) 
])

@app.callback(
    [Output('cytoscape-container', 'children'),
     Output('legend-container', 'children'),
     Output('object-dropdown', 'options')],
    [Input('Update_Graph-Button', 'n_clicks')],
    [State('subject-dropdown', 'value'),
     State('predicate-dropdown', 'value'),
     State('object-dropdown', 'value'),
     State('layout-dropdown', 'value')],
    prevent_initial_call=True
)
def update_cytoscape_graph_and_legend(n_clicks, selected_subjects, selected_predicates, selected_objects, layout_algorithm):
    if not selected_subjects and not selected_predicates and not selected_objects:
        elements = cyto_elements
    if not selected_objects:
        selected_objects=[' ']
    if not selected_subjects:
        selected_subjects=[' ']
    if not selected_predicates:
        selected_predicates=[' ']
    if selected_predicates!=[' '] or selected_objects!=[' '] or selected_subjects!=[' ']:
        filtered_elements = []
        selected_subjects = [URIRef(subject) for subject in selected_subjects]
        selected_objects = [Literal(obj) for obj in selected_objects] if selected_objects else None
        for node in nx_graph.nodes():
            if node in selected_subjects or node in selected_objects:
                filtered_elements.append({'data': {'id': node, 'label': node.split('#')[-1]}})
        if selected_predicates != [' ']:
            edges = nx.get_edge_attributes(nx_graph, 'label')
            for edge, label in edges.items():  
                source, target = edge  
                if label in selected_predicates:
                    filtered_elements.append({'data': {'id': source, 'label': source}})
                    filtered_elements.append({'data': {'id': target, 'label': target}})
                    filtered_elements.append({'data': {'source': source, 'target': target, 'label': label}})
        else:
            for edge in nx_graph.edges():
                source, target = edge
                if source in selected_subjects or source in selected_objects:
                    filtered_elements.append({'data': {'id': target, 'label': target.split('#')[-1]}})
                    filtered_elements.append({'data': {'source': source, 'target': target, 'label': nx_graph.edges[edge]['label']}})            
        elements = filtered_elements
    
    layout = {'name': layout_algorithm}
    
    legend_items = [
        html.Div([
            html.Div(style={'width': '20px', 'height': '20px', 'background-color': color, 'display': 'inline-block'}),
            html.Span(f" {predicate}", style={'margin-left': '5px'})
        ]) for predicate, color in predicate_color_map.items()
    ]
    
    object_options = [{'label': obj, 'value': obj} for obj in g.objects(None, URIRef(selected_predicates[0]))] if selected_predicates else []
    
    return (
        html.Div([
            cyto.Cytoscape(
                id='cytoscape-graph',
                layout=layout,
                style={'width': '90%', 'height': '1500px'},
                elements=elements,
                stylesheet=cyto_stylesheet
            )
        ]),
        legend_items,
        object_options
    )

if __name__ == '__main__':
    app.run_server(debug=True)
