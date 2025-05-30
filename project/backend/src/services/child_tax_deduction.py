import pandas as pd

def load_tax_table(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        sep='\t' if '\t' in open(csv_path).readline() else ',',
        dtype={'code': str}
    )
    df.set_index('code', inplace=True)
    return df

class TaxSimulator:
    """
    Simula la recaudación de impuestos indirectos tras aplicar deducciones por hijo.

    Args:
      infer_fn: función de inferencia que recibe inputs: list[float] y devuelve dict[int, float]
      feature_cols: lista de nombres de las variables de entrada en el mismo orden que inputs
      output_codes: lista de códigos de categoría (int) en el mismo orden de las predicciones
      tax_csv_path: ruta al CSV/TSV con tipos de IVA, IGIC e IPSI.
    """
    def __init__(self, infer_fn, feature_cols, output_codes, tax_csv_path='datamarts/tax_datamart_2025.tsv'):
        self.infer = infer_fn
        self.feature_cols = feature_cols
        # Mantenemos los códigos como enteros
        self.output_codes = list(output_codes)
        self.tax_df = load_tax_table(tax_csv_path)

    def simulate(self, inputs, deduction_per_child, ccaa):
        # 1) Inferencia original
        orig = self.infer(inputs)

        # 2) Calcular nº de hijos
        idx_fem = self.feature_cols.index('NNINO_FEM')
        idx_mas = self.feature_cols.index('NNINO_MAS')
        num_children = int(inputs[idx_fem]) + int(inputs[idx_mas])

        print(idx_fem, idx_mas, num_children)

        # 3) Ajustar IMPEXAC
        idx_impexac = self.feature_cols.index('IMPEXAC')
        adjusted_inputs = inputs.copy()
        adjusted_inputs[idx_impexac] += deduction_per_child * num_children

        # 4) Inferencia ajustada
        adj = self.infer(adjusted_inputs)

        # 5) Calcular impacto y recaudación
        impact = {}
        revenue_by_category = {}

        for code in self.output_codes:
            orig_val = orig.get(code, 0.0)
            adj_val  = adj.get(code, 0.0)
            delta    = adj_val - orig_val
            impact[code] = delta

            code_str = str(code)  # para indexar tax_df
            # Selección de tasa según CCAA
            if ccaa == 5:          # Canarias → IGIC
                rate = float(self.tax_df.at[code_str, 'igic']) / 100.0
            elif ccaa in (18, 19): # Ceuta/Melilla → IPSI
                rate = float(self.tax_df.at[code_str, 'ipsi']) / 100.0
            else:                  # Resto → IVA
                rate = float(self.tax_df.at[code_str, 'iva']) / 100.0

            revenue_by_category[code] = delta * rate

        total_revenue = sum(revenue_by_category.values())

        return {
            'original': orig,
            'adjusted': adj,
            'impact': impact,
            'revenue_by_category': revenue_by_category,
            'total_revenue': total_revenue,
            'num_children': num_children,
            'deduction_per_child': deduction_per_child
        }
