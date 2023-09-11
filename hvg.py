import cellxgene_census
from cellxgene_census.experimental.pp import get_highly_variable_genes, highly_variable_genes
import pandas as pd
import tiledbsoma as soma
import pdb


with cellxgene_census.open_soma(uri='/mnt/census') as census:
    hvgs_df = get_highly_variable_genes(
        census,
        organism="homo_sapiens",
        n_top_genes=3000,
        obs_value_filter="""is_primary_data == True""",
    )

    print(hvgs_df)
    #pdb.set_trace()
    hvgs_df[hvgs_df.highly_variable][[]].to_csv('hvgs.csv')

                
