"""프로젝트 흐름 이해용 폴더(PROJECT_FLOW)를 만든다: 읽는 순서대로 번호를 붙여 원본 문서를 복사한다(원본은 그대로).
보고서가 본문에 넣은 그림은 같은 폴더에 원래 이름으로 함께 복사해 링크가 깨지지 않게 한다.
FF++ 얼굴 이미지가 든 그림(s3_*_grid.png, s6_align_grid_*.png)은 약관(공개 게시 금지) 때문에 복사하지 않는다.
원본이 바뀌면 다시 실행: python code/make_project_flow.py   (파이프라인 그림은 python code/flow_figs.py)"""
import os
import shutil

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(EXP, 'PROJECT_FLOW')
R, P = 'reports', 'paper'
PLAN = {
    '01_설계와_규칙': [('HANDOFF.md', '01_HANDOFF.md'), ('PROTOCOL.md', '02_PROTOCOL.md')],
    '02_기능추가실험_S0-S7': [(f'{R}/S7_report.md', '01_S7_report.md'), (f'{R}/s7_tables.md', '02_s7_tables.md'),
                          (f'{R}/s7_fig_ablation.png', None), (f'{R}/s7_fig_curves_100.png', None), (f'{R}/s7_fig_datasize.png', None),
                          (f'{R}/s8_fig_curves.png', None), (f'{R}/s8_fig_compare.png', None), (f'{R}/s9_fig_compare.png', None),
                          (f'{R}/s10_fig_compare.png', None)],
    '03_후속검증_S8-S10': [(f'{R}/s8_tables.md', '01_s8_tables.md'), (f'{R}/s9_tables.md', '02_s9_tables.md'),
                        (f'{R}/s10_tables.md', '03_s10_tables.md'), (f'{R}/s10_bootstrap.md', '04_s10_bootstrap.md')],
    '04_개선1차_TASKS4': [('TASKS4.md', '01_TASKS4.md'), (f'{R}/t3_best_combo.md', '02_t3_best_combo.md'),
                       (f'{R}/t4_improvements.md', '03_t4_improvements.md'), (f'{R}/t4_fig_anchor.png', None),
                       (f'{R}/t4_ensemble.md', '04_t4_ensemble.md'), (f'{R}/t4_fusion.md', '05_t4_fusion.md'),
                       (f'{R}/t4_results.md', '06_t4_results.md')],
    '05_최종모델_TASKS5': [('TASKS5.md', '01_TASKS5.md'), (f'{R}/t5_final.md', '02_t5_final.md'), (f'{R}/t5_fig_merge.png', None),
                        (f'{R}/t5_final_tables.md', '03_t5_final_tables.md'), (f'{R}/t5_merge.md', '04_t5_merge.md'),
                        (f'{R}/t5_calib.md', '05_t5_calib.md'), (f'{R}/t5_beyond_auc.md', '06_t5_beyond_auc.md'),
                        (f'{R}/t5_beyond_tables.md', '07_t5_beyond_tables.md')],
    '06_추가탐색_TASKS6': [('TASKS6.md', '01_TASKS6.md'), (f'{R}/t6_final.md', '02_t6_final.md'),
                        (f'{R}/t6_final_tables.md', '03_t6_final_tables.md'), (f'{R}/t6_agg.md', '04_t6_agg.md')],
    '07_논문_발표': [(f'{P}/draft_v2.md', '01_draft_v2_논문초안_최종.md'), (f'{P}/final_presentation_outline.md', '02_최종발표_개요.md'),
                 (f'{P}/presentation_highlights.md', '03_발표_수치와_표현.md'), (f'{P}/value_assessment.md', '04_가치평가.md'),
                 (f'{P}/draft_v1.md', '05_draft_v1_논문초안_분석중심.md')],
    '08_진행기록_참고': [('PROGRESS.md', '01_PROGRESS.md'), ('GUIDE.md', '02_GUIDE.md'), ('registry.csv', '03_registry.csv')],
}
BLOCK = ('s3_sbi_grid.png', 's3_seghead_grid.png', 's6_align_grid_test_probe3.png')


def main():
    n = 0
    for folder, items in PLAN.items():
        d = os.path.join(OUT, folder)
        os.makedirs(d, exist_ok=True)
        for src, dst in items:
            assert os.path.basename(src) not in BLOCK, src
            shutil.copy2(os.path.join(EXP, src), os.path.join(d, dst or os.path.basename(src)))
            n += 1
    print(f'{n}개 파일 복사 → {OUT}')


if __name__ == '__main__':
    main()
