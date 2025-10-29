import pandas as pd
import numpy as np

import dash
from dash import Dash, html, dcc, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------
# CONFIG
# -------------------------------
CSV_PATH = 'Finance_data.csv'  # Put the CSV in the same folder as this app
THEME = dbc.themes.LUX          # Try CYBORG or MATERIA if you want dark

# Numeric investment columns in the dataset (safeguard for typos)
INV_COLS = [
    'Mutual_Funds', 'Equity_Market', 'Debentures', 'Government_Bonds',
    'Fixed_Deposits', 'PPF', 'Gold'
]

# -------------------------------
# DATA
# -------------------------------
df = pd.read_csv(CSV_PATH)

# Some datasets have a misspelled column for stock market; we won't rely on it
# but keep a safe alias if needed.
if 'Stock_Marktet' in df.columns and 'Stock_Market' not in df.columns:
    df.rename(columns={'Stock_Marktet': 'Stock_Market'}, inplace=True)

# Canonical categorical columns used below (guard against missing)
for col, fallback in {
    'gender': 'Unknown',
    'Investment_Avenues': 'Unknown',
    'Objective': 'Unknown',
    'Purpose': 'Unknown',
    'Duration': 'Unknown',
    'Source': 'Unknown',
}.items():
    if col not in df.columns:
        df[col] = fallback

if 'age' not in df.columns:
    df['age'] = np.nan

# Ensure investment columns exist
for c in INV_COLS:
    if c not in df.columns:
        df[c] = 0

# Precompute overall min/max for age slider
age_min = int(np.nanmin(df['age'])) if df['age'].notna().any() else 18
age_max = int(np.nanmax(df['age'])) if df['age'].notna().any() else 60

# Helper: aggregate investment scores per avenue
def avenue_scores(frame: pd.DataFrame) -> pd.DataFrame:
    totals = (
        frame[INV_COLS]
        .sum(numeric_only=True)
        .reset_index()
        .rename(columns={'index': 'Avenue', 0: 'Score'})
    )
    totals.columns = ['Avenue', 'Score']
    totals.sort_values('Score', ascending=False, inplace=True)
    return totals

# Helper: top mode with safe fallback
def safe_mode(series: pd.Series, default='—'):
    try:
        m = series.mode(dropna=True)
        return m.iloc[0] if not m.empty else default
    except Exception:
        return default

# -------------------------------
# APP
# -------------------------------
app: Dash = dash.Dash(__name__, external_stylesheets=[THEME])
app.title = 'Finance Analytics Dashboard'

# ---------- Filters (left column)
filter_card = dbc.Card([
    dbc.CardHeader(html.H5('Filters', className='mb-0')),
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                dbc.Label('Gender'),
                dcc.Dropdown(
                    id='f-gender',
                    options=[{'label': g, 'value': g} for g in sorted(df['gender'].dropna().unique())],
                    value=sorted(df['gender'].dropna().unique()),
                    multi=True,
                    placeholder='All genders'
                )
            ], width=12),
        ], className='g-2'),
        dbc.Row([
            dbc.Col([
                dbc.Label('Objective'),
                dcc.Dropdown(
                    id='f-obj',
                    options=[{'label': o, 'value': o} for o in sorted(df['Objective'].dropna().unique())],
                    value=sorted(df['Objective'].dropna().unique()),
                    multi=True,
                    placeholder='All objectives'
                )
            ], width=12),
        ], className='g-2'),
        dbc.Row([
            dbc.Col([
                dbc.Label('Duration'),
                dcc.Dropdown(
                    id='f-duration',
                    options=[{'label': d, 'value': d} for d in sorted(df['Duration'].dropna().unique())],
                    value=sorted(df['Duration'].dropna().unique()),
                    multi=True,
                    placeholder='All durations'
                )
            ], width=12),
        ], className='g-2'),
        dbc.Row([
            dbc.Col([
                dbc.Label('Age Range'),
                dcc.RangeSlider(id='f-age', min=age_min, max=age_max, step=1, value=[age_min, age_max],
                                tooltip={"placement": "bottom", "always_visible": False})
            ], width=12)
        ], className='mt-3')
    ])
], className='mb-3')

# ---------- KPI cards (top row center)

def kpi_card(id_value: str, title: str) -> dbc.Card:
    return dbc.Card([
        dbc.CardBody([
            html.Div(title, className='text-muted small'),
            html.H3(id=id_value, className='mb-0 fw-bold')
        ])
    ], className='shadow-sm')

kpi_row = dbc.Row([
    dbc.Col(kpi_card('kpi-respondents', 'Total Respondents'), md=3),
    dbc.Col(kpi_card('kpi-avg-age', 'Average Age'), md=3),
    dbc.Col(kpi_card('kpi-top-avenue', 'Top Investment Avenue'), md=3),
    dbc.Col(kpi_card('kpi-top-objective', 'Top Objective'), md=3),
], className='g-3')

# ---------- Charts (HR-style layout)
# Left column (like donut + bar)
left_col = dbc.Col([
    dbc.Card([
        dbc.CardHeader('Distribution by Investment Avenue'),
        dbc.CardBody([
            dcc.Graph(id='g-avenue-pie', config={'displayModeBar': False})
        ])
    ], className='mb-3 shadow-sm'),

    dbc.Card([
        dbc.CardHeader('Gender Split'),
        dbc.CardBody([
            dcc.Graph(id='g-gender-bar', config={'displayModeBar': False})
        ])
    ], className='mb-3 shadow-sm'),
], md=4)

# Middle column (table + age distribution)
mid_col = dbc.Col([
    dbc.Card([
        dbc.CardHeader('Investment Scores by Avenue'),
        dbc.CardBody([
            dash_table.DataTable(
                id='tbl-avenue-scores',
                columns=[{'name': 'Avenue', 'id': 'Avenue'}, {'name': 'Score', 'id': 'Score'}],
                style_cell={'padding': '6px', 'fontSize': 13},
                style_header={'fontWeight': '700'},
                sort_action='native',
                page_size=7
            )
        ])
    ], className='mb-3 shadow-sm'),

    dbc.Card([
        dbc.CardHeader('Age Distribution'),
        dbc.CardBody([
            dcc.Graph(id='g-age-hist', config={'displayModeBar': False})
        ])
    ], className='mb-3 shadow-sm'),
], md=4)

# Right column (bars)
right_col = dbc.Col([
    dbc.Card([
        dbc.CardHeader('Objectives Distribution'),
        dbc.CardBody([
            dcc.Graph(id='g-objective-bar', config={'displayModeBar': False})
        ])
    ], className='mb-3 shadow-sm'),

    dbc.Card([
        dbc.CardHeader('Duration Distribution'),
        dbc.CardBody([
            dcc.Graph(id='g-duration-bar', config={'displayModeBar': False})
        ])
    ], className='mb-3 shadow-sm'),

    dbc.Card([
        dbc.CardHeader('Information Source (Top)'),
        dbc.CardBody([
            dcc.Graph(id='g-source-bar', config={'displayModeBar': False})
        ])
    ], className='mb-3 shadow-sm'),
], md=4)

# ---------- App Layout
app.layout = dbc.Container([
    html.H3('Financial Analytics Dashboard', className='mt-3 mb-2 fw-bold'),
    dbc.Row([
        dbc.Col(filter_card, md=3),
        dbc.Col([
            kpi_row,
            dbc.Row([left_col, mid_col, right_col], className='mt-1')
        ], md=9)
    ], className='g-3')
], fluid=True)

# -------------------------------
# CALLBACKS
# -------------------------------

def apply_filters(frame: pd.DataFrame, genders, objs, durs, age_range):
    f = frame.copy()
    if genders:
        f = f[f['gender'].isin(genders)]
    if objs:
        f = f[f['Objective'].isin(objs)]
    if durs:
        f = f[f['Duration'].isin(durs)]
    if age_range and 'age' in f.columns:
        f = f[(f['age'] >= age_range[0]) & (f['age'] <= age_range[1])]
    return f

@app.callback(
    Output('kpi-respondents', 'children'),
    Output('kpi-avg-age', 'children'),
    Output('kpi-top-avenue', 'children'),
    Output('kpi-top-objective', 'children'),
    Output('g-avenue-pie', 'figure'),
    Output('g-gender-bar', 'figure'),
    Output('tbl-avenue-scores', 'data'),
    Output('g-age-hist', 'figure'),
    Output('g-objective-bar', 'figure'),
    Output('g-duration-bar', 'figure'),
    Output('g-source-bar', 'figure'),
    Input('f-gender', 'value'),
    Input('f-obj', 'value'),
    Input('f-duration', 'value'),
    Input('f-age', 'value')
)
def update_dashboard(genders, objs, durs, age_range):
    filtered = apply_filters(df, genders, objs, durs, age_range)

    # KPIs
    total_resp = len(filtered)
    avg_age = round(filtered['age'].mean(), 1) if filtered['age'].notna().any() else '—'

    # Top avenue by summed score across numeric columns
    scores = avenue_scores(filtered)
    top_avenue = scores.iloc[0]['Avenue'] if not scores.empty else '—'
    top_obj = safe_mode(filtered['Objective'])

    # Pie: Investment_Avenues distribution (categorical choice)
    pie_df = filtered['Investment_Avenues'].value_counts(dropna=False).reset_index()
    pie_df.columns = ['Avenue', 'Count']
    fig_pie = px.pie(pie_df, names='Avenue', values='Count', hole=0.45)
    fig_pie.update_layout(margin=dict(l=0, r=0, t=0, b=0))

    # Gender split bar (horizontal)
    gdf = filtered['gender'].value_counts(dropna=False).reset_index()
    gdf.columns = ['gender', 'count']
    fig_gender = px.bar(gdf, y='gender', x='count', orientation='h', text='count')
    fig_gender.update_layout(margin=dict(l=0, r=0, t=0, b=0), yaxis_title='', xaxis_title='')

    # Table data
    table_data = scores.to_dict('records')

    # Age histogram (line-like by using histogram with stepped line)
    if filtered['age'].notna().any():
        fig_age = px.histogram(filtered, x='age', nbins=min(10, max(5, int(np.sqrt(len(filtered))))),
                               histnorm='')
        fig_age.update_layout(margin=dict(l=0, r=0, t=0, b=0), xaxis_title='Age', yaxis_title='Count')
    else:
        fig_age = go.Figure()
        fig_age.update_layout(margin=dict(l=0, r=0, t=0, b=0), xaxis_visible=False, yaxis_visible=False,
                              annotations=[dict(text='No age data', x=0.5, y=0.5, showarrow=False)])

    # Objectives bar
    obj_df = filtered['Objective'].value_counts(dropna=False).reset_index()
    obj_df.columns = ['Objective', 'Count']
    fig_obj = px.bar(obj_df, x='Objective', y='Count', text='Count')
    fig_obj.update_layout(margin=dict(l=0, r=0, t=0, b=0), xaxis_title='', yaxis_title='')

    # Duration bar
    dur_df = filtered['Duration'].value_counts(dropna=False).reset_index()
    dur_df.columns = ['Duration', 'Count']
    fig_dur = px.bar(dur_df, x='Duration', y='Count', text='Count')
    fig_dur.update_layout(margin=dict(l=0, r=0, t=0, b=0), xaxis_title='', yaxis_title='')

    # Source bar (top categories)
    source_df = (
        filtered['Source']
        .value_counts(dropna=False)
        .head(10)
        .sort_values(ascending=True)
        .reset_index()
    )
    source_df.columns = ['Source', 'Count']
    fig_source = px.bar(source_df, y='Source', x='Count', orientation='h', text='Count')
    fig_source.update_layout(margin=dict(l=0, r=0, t=0, b=0), xaxis_title='', yaxis_title='')

    return (
        f"{total_resp}",
        f"{avg_age}",
        top_avenue,
        top_obj,
        fig_pie,
        fig_gender,
        table_data,
        fig_age,
        fig_obj,
        fig_dur,
        fig_source,
    )

# -------------------------------
# MAIN
# -------------------------------
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=True)