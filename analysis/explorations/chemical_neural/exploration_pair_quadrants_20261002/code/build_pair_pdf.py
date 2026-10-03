"""Build the requested per-pair exploration PDF without category summaries.

Run with the Codex bundled Python (ReportLab, pandas, numpy, pypdf). Existing
distances and quartile assignments are reused, not recomputed or refitted.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.colors import HexColor
from pypdf import PdfReader

from pair_pdf_pages import draw_pair_page, TOP_N

CATEGORIES = ('Cnear_Nnear','Cfar_Nfar','Cnear_Nfar','Cfar_Nnear')
LABELS = {'Cnear_Nnear':'Neural near / Chemical near',
          'Cfar_Nfar':'Neural far / Chemical far',
          'Cnear_Nfar':'Neural far / Chemical near',
          'Cfar_Nnear':'Neural near / Chemical far'}


def recurrent_features(pairs, chemical):
    """Count membership in pair-specific top18 lists, preserving feature IDs.

    Red bold names mean a feature appears in the top18 of at least one pair in each of
    at least two distance categories. It does not imply independent replication.
    """
    counts = pd.DataFrame(0, index=chemical.columns, columns=CATEGORIES, dtype=int)
    for row in pairs.itertuples():
        delta = np.abs(chemical.loc[row.strain_a].to_numpy() - chemical.loc[row.strain_b].to_numpy())
        selected = np.argsort(-delta, kind='stable')[:TOP_N]
        counts.loc[chemical.columns[selected], row.category] += 1
    counts.index.name = 'feature'
    counts['n_categories'] = counts.gt(0).sum(axis=1)
    counts['n_pair_appearances'] = counts[list(CATEGORIES)].sum(axis=1)
    counts['bold'] = counts.n_categories.ge(2)
    return counts


def load_data(repo, out):
    table = out/'tables'
    pairs = pd.read_csv(table/'pair_catalogue.csv')
    chemical = pd.read_csv(table/'sample_chemical_log2fc.csv',index_col=0)
    neural = pd.read_csv(table/'sample_neural_coefficients.csv',index_col=0)
    arrays = np.load(table/'pair_arrays.npz')
    # Use each sample's own original-report status for hollow markers.
    source = repo/'reports/population_first_20260930/tables'
    raw = pd.read_csv(source/'aligned_chemical_report_values_all.csv',index_col=0).loc[chemical.index,chemical.columns]
    data = dict(chemical=chemical,neural=neural,reported=raw.notna(),
        meta=pd.read_csv(table/'feature_metadata.csv',index_col=0).loc[chemical.columns],
        taxonomy=pd.read_csv(source/'aligned_taxonomy_paired.csv',index_col=0),
        chemical_limits=(-24.,24.),neural_limits=(-.5,3.))
    assert arrays['features'].tolist() == chemical.columns.tolist()
    assert arrays['cells'].tolist() == neural.columns.tolist()
    assert arrays['pair_ids'].tolist() == pairs.pair_id.tolist()
    a=chemical.index.get_indexer(pairs.strain_a);b=chemical.index.get_indexer(pairs.strain_b)
    assert np.array_equal(data['reported'].to_numpy()[a]&data['reported'].to_numpy()[b],arrays['both_reported'])
    rows=[]
    for cat in CATEGORIES:
        subset=pairs[pairs.category.eq(cat)].sort_values(['chemical','neural','strain_a','strain_b'])
        rows.extend(subset.to_dict('records'))
    ordered=pd.DataFrame(rows)
    ordered.insert(0,'pdf_page',np.arange(2,len(ordered)+2))
    assert len(ordered)==1478 and ordered.pair_id.is_unique
    recurrence = recurrent_features(ordered,chemical)
    data['chemical_recurrence'] = recurrence
    data['bold_chemicals'] = set(recurrence.index[recurrence.bold])
    return ordered, data


def register_fonts(repo):
    fonts=repo/'.pixi/envs/default/lib/python3.11/site-packages/matplotlib/mpl-data/fonts/ttf'
    pdfmetrics.registerFont(TTFont('AtlasSans',str(fonts/'DejaVuSans.ttf')))
    pdfmetrics.registerFont(TTFont('AtlasBold',str(fonts/'DejaVuSans-Bold.ttf')))


def draw_guide(pdf, pairs, params):
    width,height=landscape(A4)
    pdf.setFillColor(HexColor('#243746'))
    pdf.setFont('AtlasBold',23)
    pdf.drawString(42,height-58,'Pair-by-pair neural and chemical exploration')
    pdf.setFont('AtlasSans',11)
    pdf.drawString(42,height-82,f'{len(pairs):,} sample pairs | one pair per page | 106 samples | 2 October 2026')
    pdf.setFont('AtlasSans',10)
    pdf.drawString(42,height-121,'Navigate by section or sample-pair bookmarks. Each pair is plotted independently.')
    y=height-165
    for cat in CATEGORIES:
        z=pairs[pairs.category.eq(cat)]
        text=f'{LABELS[cat]}    {len(z)} pairs    pages {z.pdf_page.min()}-{z.pdf_page.max()}'
        pdf.setFont('AtlasBold',12);pdf.drawString(54,y,text)
        pdf.linkRect('',cat,(48,y-5,width-50,y+18),relative=0,thickness=0)
        y-=34
    y-=14
    lines=[
        'Upper panels: signed neural coefficients; normalized response profiles; all 380 chemical features.',
        f'Lower panel: the {TOP_N} largest absolute chemical differences for this specific pair (full feature set).',
        'Red bold names: top-18 in at least two distance categories; red dots/bars identify the second sample.',
        'Open chemical markers mean an original-report value is missing; they do not establish absence.',
        'Neural coefficients: existing individual SNR >= 0.5 representation, 13 cells, 0-40 s templates.',
        'Coefficient signs are relative to each cell template; screened zeros do not establish no response.',
        'Chemical values: existing log2FC relative to medium reference; not absolute concentrations.',
        'Distances: neural = 1 - cosine; chemical = RMS log2FC difference. No fit or bootstrap was rerun.',
        'Near/far: bottom/top 25% of all 5,565 pair distances. Intermediate pairs are outside this atlas.',
        'Within each section, pairs are ordered by chemical distance, then neural distance and IDs.',
        'Reference IDs and valid-bootstrap-draw fractions are context, not biological labels or confidence.',
        'Names are report annotations. Samples recur across pairs; pages are not independent replicates.',
    ]
    pdf.setFont('AtlasSans',9)
    for line in lines:pdf.drawString(54,y,line);y-=18
    pdf.setFillColor(HexColor('#606b75'));pdf.setFont('AtlasSans',8)
    pdf.drawString(42,26,'Exploratory inspection only. No category averages, representative-pair selection, or Figure 5 conclusion.')
    pdf.drawRightString(width-42,26,'1')


def build_pdf(repo, out, output_pdf, preview=False):
    repo,out,output_pdf=Path(repo),Path(out),Path(output_pdf)
    register_fonts(repo)
    ordered,data=load_data(repo,out)
    params=json.loads((out/'parameters.json').read_text())
    if preview:
        # First, median and last pair per category plus maximum-label case.
        positions=[]
        for cat in CATEGORIES:
            ix=np.flatnonzero(ordered.category.eq(cat))
            positions.extend([ix[0],ix[len(ix)//2],ix[-1]])
        selected=ordered.iloc[sorted(set(positions))].copy()
    else:selected=ordered
    output_pdf.parent.mkdir(parents=True,exist_ok=True)
    pdf=canvas.Canvas(str(output_pdf),pagesize=landscape(A4),pageCompression=1)
    pdf.setTitle('Pair-by-pair neural and chemical exploration')
    pdf.setAuthor('Bacteria analysis - exploratory figures')
    pdf.bookmarkPage('guide');pdf.addOutlineEntry('Reading guide','guide',level=0,closed=False)
    draw_guide(pdf,ordered,params);pdf.showPage()
    seen=set()
    for i,(_,row) in enumerate(selected.iterrows(),start=2):
        cat=row.category
        if cat not in seen:
            pdf.bookmarkPage(cat);pdf.addOutlineEntry(LABELS[cat],cat,level=0,closed=True);seen.add(cat)
        key=row.pair_id
        pdf.bookmarkPage(key);pdf.addOutlineEntry(f'{row.strain_a} / {row.strain_b}',key,level=1,closed=False)
        draw_pair_page(pdf,row,data,i)
        pdf.showPage()
        if not preview and (i%100==0):print(f'PDF pages {i}/{len(selected)+1}',flush=True)
    pdf.save()
    reader=PdfReader(str(output_pdf))
    assert len(reader.pages)==len(selected)+1
    if not preview:
        data['chemical_recurrence'].to_csv(out/'tables/pair_pdf_chemical_recurrence.csv')
        ordered[['pdf_page','category','pair_id','strain_a','strain_b','chemical','neural',
                 'reference_a','reference_b','bootstrap_valid_fraction']].to_csv(out/'tables/pair_pdf_index.csv',index=False)
        result=dict(pdf=str(output_pdf.resolve()),n_pages=len(reader.pages),n_pair_pages=len(ordered),
            all_selected_pairs_included=True,one_pair_per_page=True,no_category_averages=True,
            chemical_panel_features=380,neural_panel_cells=13,pair_specific_top_features=TOP_N,
            bold_rule='Appears in pair-specific top18 in at least two distinct distance categories',
            n_bold_chemicals=len(data['bold_chemicals']),
            red_name_rule='Appears in pair-specific top18 in at least two distinct distance categories',
            n_red_chemicals=len(data['bold_chemicals']),
            recurrent_name_color='#C64245',
            recurrence_names_remain_bold=True,
            global_chemical_limits=data['chemical_limits'],global_neural_limits=data['neural_limits'],
            sha256=hashlib.sha256(output_pdf.read_bytes()).hexdigest(),
            source_hashes=json.loads((out/'parameters.json').read_text())['source_sha256'])
        (out/'pair_pdf_verification.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:result[k] for k in ['pdf','n_pages','n_pair_pages','sha256']},indent=2))
    return output_pdf


if __name__=='__main__':
    import sys
    out=Path(__file__).resolve().parents[1]
    repo=out.parents[1]
    preview='--preview' in sys.argv
    path=(repo/'tmp/pdfs/pair_exploration_preview.pdf' if preview else
          repo/'output/pdf/neural_chemical_pairs_exploration.pdf')
    build_pdf(repo,out,path,preview=preview)
