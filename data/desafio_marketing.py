import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler

# Configuración de la página (tema claro por defecto)
st.set_page_config(
    page_title="Dashboard Ejecutivo de Marketing",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Dashboard Interactivo de Marketing")
st.markdown("""
Este dashboard permite explorar datos de campañas de marketing de forma interactiva. 
Utiliza los filtros y selectores en la barra lateral para personalizar tu análisis.
""")

@st.cache_data(ttl=3600)
def load_data():
    try:
        df = pd.read_csv("data/desafio_marketing_limpio.csv")
        # Normalizar a minúsculas para evitar problemas de mayúsculas/minúsculas
        for col in ['type', 'channel', 'target_audience', 'theme']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower()
        # Quitar "redes sociales" de audiencia
        if 'target_audience' in df.columns:
            df = df[~df['target_audience'].str.contains("redes sociales", case=False, na=False)]
        # Quitar "b2b" de tipo de campaña
        if 'type' in df.columns:
            df = df[~df['type'].str.contains("b2b", case=False, na=False)]
        st.sidebar.success("Datos cargados correctamente de data/")
        return df
    except Exception as e:
        st.error(f"Error al cargar datos: {str(e)}")
        return None

roi_colors = {
    'pérdida': 'red',
    'bajo': 'orange',
    'medio': 'blue',
    'alto': 'green',
    'no clasificado': 'gray'
}

def ensure_positive(values, min_size=3):
    if isinstance(values, (pd.Series, np.ndarray, list)):
        return np.maximum(np.abs(values), min_size)
    else:
        return max(abs(values), min_size)

df = load_data()
if df is None or df.empty:
    st.stop()

numeric_cols = ['budget', 'revenue', 'roi', 'conversion_rate', 'net_profit']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
if 'net_profit' not in df.columns and 'revenue' in df.columns and 'budget' in df.columns:
    df['net_profit'] = df['revenue'] - df['budget']

if 'roi' in df.columns:
    df['roi'] = pd.to_numeric(df['roi'], errors='coerce')
    conditions = [
        (df['roi'] < 0),
        (df['roi'] >= 0) & (df['roi'] < 0.5),
        (df['roi'] >= 0.5) & (df['roi'] < 1),
        (df['roi'] >= 1)
    ]
    choices = ['pérdida', 'bajo', 'medio', 'alto']
    df['roi_categoria'] = np.select(conditions, choices, default='no clasificado')

st.sidebar.header("🎯 Filtros")
for col in ['type', 'channel', 'target_audience', 'theme']:
    if col in df.columns:
        df[col] = df[col].astype(str)

selected_types = st.sidebar.multiselect(
    "Tipos de campaña", options=sorted(df['type'].unique()), default=sorted(df['type'].unique())
) if 'type' in df.columns else []
selected_channels = st.sidebar.multiselect(
    "Canales", options=sorted(df['channel'].unique()), default=sorted(df['channel'].unique())
) if 'channel' in df.columns else []
selected_audiences = st.sidebar.multiselect(
    "Audiencia objetivo", options=sorted(df['target_audience'].unique()), default=sorted(df['target_audience'].unique())
) if 'target_audience' in df.columns else []
selected_themes = st.sidebar.multiselect(
    "Temas", options=sorted(df['theme'].unique()), default=sorted(df['theme'].unique())
) if 'theme' in df.columns else []

roi_values = pd.to_numeric(df['roi'], errors='coerce').dropna()
min_roi, max_roi = float(roi_values.min()), float(roi_values.max())
roi_range = st.sidebar.slider("Rango de ROI", min_value=min_roi, max_value=max_roi, value=(min_roi, max_roi), step=0.05)

budget_values = pd.to_numeric(df['budget'], errors='coerce').dropna()
min_budget, max_budget = float(budget_values.min()), float(budget_values.max())
budget_range = st.sidebar.slider("Rango de presupuesto", min_value=min_budget, max_value=max_budget, value=(min_budget, max_budget), step=1000.0)

filtered_df = df.copy()
if selected_types: filtered_df = filtered_df[filtered_df['type'].isin(selected_types)]
if selected_channels: filtered_df = filtered_df[filtered_df['channel'].isin(selected_channels)]
if selected_audiences: filtered_df = filtered_df[filtered_df['target_audience'].isin(selected_audiences)]
if selected_themes: filtered_df = filtered_df[filtered_df['theme'].isin(selected_themes)]
filtered_df = filtered_df[
    (filtered_df['roi'] >= roi_range[0]) & (filtered_df['roi'] <= roi_range[1]) &
    (filtered_df['budget'] >= budget_range[0]) & (filtered_df['budget'] <= budget_range[1])
]

st.sidebar.metric("Campañas seleccionadas", len(filtered_df))

main_tabs = st.tabs([
    "📊 Dashboard", 
    "📈 Análisis por Canal y Tema", 
    "🔗 Correlaciones", 
    "🔍 Análisis Interactivo", 
    "💡 Análisis por Tema"
])

# --- Dashboard General ---
with main_tabs[0]:
    st.header("Resumen Ejecutivo")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Campañas", len(filtered_df))
    col2.metric("Presupuesto Total", f"${filtered_df['budget'].sum():,.0f}")
    col3.metric("Ingresos Totales", f"${filtered_df['revenue'].sum():,.0f}")
    roi_mean = filtered_df['roi'].mean() if not filtered_df['roi'].empty else 0
    col4.metric("ROI Promedio", f"{roi_mean:.2f}")

    st.subheader("Distribución de ROI y Presupuesto")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(filtered_df, x="roi", color="roi_categoria", nbins=30, color_discrete_map=roi_colors, title="Distribución de ROI")
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**Insight:** El ROI promedio es de {roi_mean:.2f}. La mayoría de campañas se concentran en la categoría '{filtered_df['roi_categoria'].mode()[0]}'.")
    with c2:
        fig = px.histogram(filtered_df, x="budget", color="roi_categoria", nbins=30, color_discrete_map=roi_colors, title="Distribución de Presupuesto")
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**Insight:** El presupuesto medio por campaña es ${filtered_df['budget'].mean():,.0f}. Se observa mayor inversión en campañas con ROI '{filtered_df.loc[filtered_df['budget'].idxmax(), 'roi_categoria']}'.")

    st.subheader("Top 10 Campañas por ROI")
    top_campaigns = filtered_df.sort_values('roi', ascending=False).head(10)
    fig = px.bar(top_campaigns, x='roi', y='campaign_name', orientation='h', color='roi', color_continuous_scale='Viridis', text='roi')
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Insight:** Estas campañas destacan por su alto ROI. Analizar sus características puede ayudar a replicar el éxito en futuras campañas.")

    st.markdown("""
    ---
    ### Conclusión Ejecutiva
    El análisis global muestra que la rentabilidad media es sólida, pero existe margen de mejora en la asignación de presupuesto. Se recomienda priorizar los canales y temas con mejor desempeño y analizar las campañas top para identificar factores de éxito.
    """)

# --- Análisis por Canal y Tema ---
with main_tabs[1]:
    st.header("Análisis por Canal y Tema")
    tabs = st.tabs(["Rendimiento por Canal", "Comparativa entre Canales", "Rendimiento por Tema"])
    # Rendimiento por Canal
    with tabs[0]:
        st.subheader("Rendimiento por Canal")
        channel_perf = filtered_df.groupby('channel').agg({
            'campaign_name': 'count',
            'budget': 'sum',
            'revenue': 'sum',
            'roi': 'mean',
            'conversion_rate': 'mean'
        }).reset_index()
        channel_perf.columns = ['Canal', 'Campañas', 'Presupuesto Total', 'Ingresos Totales', 'ROI Promedio', 'Tasa de Conversión']
        channel_perf['Beneficio Neto'] = channel_perf['Ingresos Totales'] - channel_perf['Presupuesto Total']
        fig = px.bar(channel_perf, x='Canal', y='ROI Promedio', color='Ingresos Totales', text='ROI Promedio', color_continuous_scale='Viridis')
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Insight:** El canal con mayor ROI promedio es el más eficiente en términos de retorno. Considera aumentar la inversión en los canales líderes.")
        st.dataframe(channel_perf, use_container_width=True)
    # Comparativa entre Canales
    with tabs[1]:
        st.subheader("Comparativa entre Canales")
        metric = st.selectbox("Métrica para comparar", ['roi', 'conversion_rate', 'revenue', 'budget', 'net_profit'])
        fig = px.box(filtered_df, x='channel', y=metric, color='channel', points="all", hover_name="campaign_name")
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**Insight:** El canal con mayor dispersión en {metric} puede tener oportunidades de optimización o riesgos de ineficiencia.")
    # Rendimiento por Tema
    with tabs[2]:
        st.subheader("Rendimiento por Tema")
        theme_perf = filtered_df.groupby('theme').agg({
            'campaign_name': 'count',
            'budget': 'sum',
            'revenue': 'sum',
            'roi': 'mean',
            'conversion_rate': 'mean'
        }).reset_index()
        theme_perf.columns = ['Tema', 'Campañas', 'Presupuesto Total', 'Ingresos Totales', 'ROI Promedio', 'Tasa de Conversión']
        theme_perf['Beneficio Neto'] = theme_perf['Ingresos Totales'] - theme_perf['Presupuesto Total']
        fig = px.bar(theme_perf, x='Tema', y='ROI Promedio', color='Ingresos Totales', text='ROI Promedio', color_continuous_scale='Viridis')
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Insight:** Los temas con mayor ROI y tasa de conversión deben ser priorizados en la estrategia de contenidos.")
        st.dataframe(theme_perf, use_container_width=True)
    st.markdown("""
    ---
    ### Conclusión
    El análisis por canal y tema revela diferencias claras en desempeño. Se recomienda enfocar recursos en los canales y temáticas con mejores métricas y explorar mejoras en los de menor rendimiento.
    """)

# --- Correlaciones ---
with main_tabs[2]:
    st.header("🔗 Mapa de Correlaciones")
    numeric_cols = filtered_df.select_dtypes(include=['number']).columns.tolist()
    corr = filtered_df[numeric_cols].corr()
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", title="Matriz de Correlación")
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Las correlaciones ayudan a detectar oportunidades y riesgos en la inversión.")

    # Insights automáticos
    st.subheader("Insights de Correlación")
    insights = []
    for i, col1 in enumerate(numeric_cols):
        for j, col2 in enumerate(numeric_cols):
            if i < j:
                val = corr.iloc[i, j]
                if abs(val) > 0.5:
                    tipo = "positiva" if val > 0 else "negativa"
                    fuerza = "fuerte" if abs(val) > 0.7 else "moderada"
                    insights.append(f"- **{col1}** y **{col2}** tienen una correlación {tipo} {fuerza} ({val:.2f})")
    if insights:
        st.markdown("\n".join(insights))
    else:
        st.info("No se encontraron correlaciones fuertes o moderadas.")

    st.markdown("""
    ---
    ### Conclusión
    Las correlaciones detectadas permiten identificar palancas clave para mejorar el ROI y la eficiencia. Se recomienda profundizar en las relaciones más fuertes para optimizar la estrategia.
    """)

# --- Análisis Interactivo ---
with main_tabs[3]:
    st.header("🔍 Análisis Interactivo")
    st.markdown("Selecciona variables y visualiza su relación de manera personalizada")
    numeric_options = filtered_df.select_dtypes(include=['number']).columns.tolist()
    friendly_names = {
        'budget': 'Presupuesto',
        'revenue': 'Ingresos',
        'roi': 'ROI',
        'conversion_rate': 'Tasa de Conversión',
        'net_profit': 'Beneficio Neto'
    }
    col1, col2 = st.columns(2)
    with col1:
        x_var = st.selectbox("Variable X:", options=numeric_options, format_func=lambda x: friendly_names.get(x, x))
    with col2:
        y_var = st.selectbox("Variable Y:", options=[col for col in numeric_options if col != x_var], format_func=lambda x: friendly_names.get(x, x))
    chart_type = st.radio("Tipo de gráfico:", ["Dispersión", "Línea", "Barra", "Área", "Histograma 2D"], horizontal=True)
    color_var = st.selectbox("Variable para color:", options=[c for c in ['theme', 'channel', 'type', 'target_audience', 'roi_categoria'] if c in filtered_df.columns])
    size_var = st.selectbox("Variable para tamaño:", options=[col for col in numeric_options if col not in [x_var, y_var]], key="size") if chart_type == "Dispersión" else None

    needed_cols = [x_var, y_var]
    if size_var:
        needed_cols.append(size_var)
    plot_df = filtered_df.dropna(subset=needed_cols)

    if not plot_df.empty:
        if chart_type == "Dispersión":
            fig = px.scatter(
                plot_df,
                x=x_var,
                y=y_var,
                color=color_var,
                size=ensure_positive(plot_df[size_var]) if size_var else None,
                hover_name="campaign_name",
                color_discrete_map=roi_colors if color_var == 'roi_categoria' else None,
                trendline="ols"
            )
        elif chart_type == "Línea":
            fig = px.line(plot_df.sort_values(x_var), x=x_var, y=y_var, color=color_var)
        elif chart_type == "Barra":
            grouped = plot_df.groupby(color_var)[y_var].mean().reset_index()
            fig = px.bar(grouped, x=color_var, y=y_var, color=color_var)
        elif chart_type == "Área":
            fig = px.area(plot_df.sort_values(x_var), x=x_var, y=y_var, color=color_var)
        elif chart_type == "Histograma 2D":
            fig = px.density_heatmap(plot_df, x=x_var, y=y_var, color_continuous_scale="Viridis", text_auto=True)
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        # Estadísticas y correlación
        st.subheader("📊 Estadísticas Descriptivas")
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"**{friendly_names.get(x_var, x_var)}**")
            st.write(plot_df[x_var].describe())
        with c2:
            st.write(f"**{friendly_names.get(y_var, y_var)}**")
            st.write(plot_df[y_var].describe())
        correlation = plot_df[x_var].corr(plot_df[y_var])
        st.info(f"**Correlación entre variables:** {correlation:.2f}")
        st.markdown(f"**Insight:** La relación entre {friendly_names.get(x_var, x_var)} y {friendly_names.get(y_var, y_var)} es {'fuerte' if abs(correlation) > 0.7 else 'moderada' if abs(correlation) > 0.5 else 'débil'} ({correlation:.2f}).")
    st.markdown("""
    ---
    ### Conclusión
    El análisis interactivo permite descubrir relaciones personalizadas entre variables clave. Utiliza esta sección para validar hipótesis y encontrar oportunidades de mejora.
    """)

# --- Análisis por Tema ---
with main_tabs[4]:
    st.header("💡 Análisis por Tema")
    if 'theme' in filtered_df.columns:
        themes = sorted(filtered_df['theme'].unique())
        selected_theme = st.selectbox("Selecciona un tema para analizar en detalle:", options=themes)
        theme_data = filtered_df[filtered_df['theme'] == selected_theme]
        if not theme_data.empty:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Campañas", f"{len(theme_data)}", f"{len(theme_data) / len(filtered_df) * 100:.1f}% del total")
            total_budget = theme_data['budget'].sum()
            c2.metric("Presupuesto Total", f"${total_budget:,.2f}", f"{total_budget / filtered_df['budget'].sum() * 100:.1f}% del total")
            avg_roi = theme_data['roi'].mean()
            all_avg_roi = filtered_df['roi'].mean()
            c3.metric("ROI Promedio", f"{avg_roi:.2f}", f"{avg_roi - all_avg_roi:.2f} vs promedio general")
            avg_conv = theme_data['conversion_rate'].mean()
            all_avg_conv = filtered_df['conversion_rate'].mean()
            c4.metric("Conversión Promedio", f"{avg_conv:.2%}", f"{(avg_conv - all_avg_conv) * 100:.2f}% vs promedio general")
            st.subheader("Distribución de ROI en el Tema")
            fig = px.histogram(theme_data, x="roi", nbins=20, color="roi_categoria", color_discrete_map=roi_colors)
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"**Insight:** El tema '{selected_theme}' destaca por un ROI promedio de {avg_roi:.2f} y una tasa de conversión de {avg_conv:.2%}.")
            st.subheader("Campañas Destacadas")
            st.dataframe(theme_data.sort_values('roi', ascending=False).head(10), use_container_width=True)
    st.markdown("""
    ---
    ### Conclusión
    El análisis por tema permite identificar áreas de contenido con mayor impacto. Se recomienda potenciar los temas con mejores métricas y analizar los de menor rendimiento para posibles ajustes.
    """)

st.sidebar.markdown("---")
st.sidebar.info("""
**Acerca de este Dashboard**

Este dashboard muestra datos de campañas de marketing para análisis ejecutivo.
Desarrollado con Streamlit y Plotly Express.
""")
st.markdown("---")
st.caption("Desarrollado para el Departamento de Estado | © 2024 Dirección de Marketing Estratégico")