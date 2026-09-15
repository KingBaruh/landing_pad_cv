"""Export submission figures from an existing runtime; never reruns detection.

python -m metrics.export_report_figures --run outputs/runtime_new_video
"""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


def export(run_dir, output_dir=None):
    run_dir = Path(run_dir)
    output = Path(output_dir) if output_dir else run_dir/'graphs'
    output.mkdir(parents=True,exist_ok=True)
    # Keep Matplotlib's disposable font cache in the project workspace.
    os.environ.setdefault('MPLCONFIGDIR',str(output.resolve()/'mpl_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with (run_dir/'metrics.csv').open(newline='',encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('Metrics CSV contains no samples.')
    report = json.loads((run_dir/'results.json').read_text(encoding='utf-8'))
    runtime = json.loads((run_dir/'runtime_summary.json').read_text(encoding='utf-8'))
    source_name = Path(str(runtime['config']['source'])).name
    original_width = runtime['original_size'][0]
    working_width = report['working_size'][0]
    ids = np.array([int(r['frame_id']) for r in rows])
    times = np.array([float(r['timestamp']) for r in rows])
    if len(ids) < 2 or np.any(np.diff(ids)<=0) or np.any(np.diff(times)<=0):
        raise ValueError('Expected at least two samples with increasing source IDs/times.')
    dt = float(np.median(np.diff(times)/np.diff(ids)))
    full_ids = np.arange(ids[0],ids[-1]+1)
    time = times[0]+(full_ids-ids[0])*dt
    selected = ids-ids[0]
    def numeric(key):
        values = np.full(len(full_ids),np.nan)
        values[selected] = [float(r[key]) if r.get(key) not in (None,'','None') else np.nan for r in rows]
        return values
    processing = numeric('processing_time_ms')
    latency = numeric('latency_ms')
    rms = numeric('reprojection_error_px')
    available = np.full(len(full_ids),np.nan)
    available[selected] = [int(r['pose_valid']=='True') for r in rows]
    limit = float(report['pose_error_limit_px'])
    missing = ~np.isin(full_ids,ids)
    accepted = (available==1) & np.isfinite(rms)
    rejected = (available==0) & np.isfinite(rms)
    count = int(np.count_nonzero(available==1))
    if count != report['pose_valid_frames'] or len(rows)!=report['processed_frames']:
        raise ValueError('CSV and results.json refer to different runs.')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'axes.titlesize':15,'axes.titleweight':'bold',
                         'savefig.facecolor':'white'})
    blue,orange,green = '#2457A7','#C15F12','#167560'
    def timing(ax):
        ax.plot(time,processing,color=blue,lw=1.1,label='Fast processing')
        ax.plot(time,latency,color=orange,lw=1.1,alpha=.8,label='Frame to result')
        ax.set(title='Processing time and result latency',ylabel='Time (ms)')
        ax.set_ylim(bottom=0)
        ax.legend(loc='upper right',frameon=False)
    def reprojection(ax):
        ax.plot(time,rms,color=blue,lw=1.3,label='Reprojection RMS')
        ax.scatter(time[rejected],rms[rejected],s=28,color=orange,zorder=3,label='Rejected Pose fit')
        ax.axhline(limit,color='#BA3434',ls='--',lw=1.4,label=f'Acceptance limit: {limit:.3f} px')
        ax.set(title='Pose reprojection error',ylabel='RMS error (working-image pixels)')
        ax.set_ylim(0,max(limit,float(np.nanmax(rms)) if np.isfinite(rms).any() else limit)*1.4)
        ax.legend(loc='upper right',frameon=False,ncol=3,fontsize=9)
    def availability(ax):
        ax.step(time,available,where='post',lw=1.4,color=green,label='Pose output')
        if missing.any():
            ax.scatter(time[missing],np.full(missing.sum(),-.12),s=22,marker='|',color='#6A6A6A',label='Unprocessed source frame')
        ax.set(title='Pose availability',ylabel='Pose status',ylim=(-.2,1.55))
        ax.set_yticks([0,1],['Unavailable','Available'])
        ax.legend(loc='upper right',frameon=False)
    descriptions = [
        ('01_processing_time',timing,'Latency starts after decode/playback scheduling; decoder and display time are excluded.'),
        ('02_reprojection_error',reprojection,'Missing estimates and unprocessed frames are gaps. A low fit error does not verify metric distance.'),
        ('03_pose_availability',availability,'Availability is a system-output measure; it is not ground-truth detection accuracy.'),
    ]
    for name,draw,note in descriptions:
        fig,ax = plt.subplots(figsize=(11,5.2))
        draw(ax)
        ax.set_xlabel('Source video time (s)')
        ax.grid(alpha=.18)
        ax.set_xlim(time[0],time[-1])
        fig.text(.08,.035,note,fontsize=9,color='#555555')
        fig.subplots_adjust(left=.12,right=.97,bottom=.19,top=.88)
        for extension in ('png','svg'):
            fig.savefig(output/f'{name}.{extension}',dpi=200)
        plt.close(fig)
    fig,axes = plt.subplots(3,1,figsize=(11,10),sharex=True)
    for ax,(_,draw,_) in zip(axes,descriptions):
        draw(ax)
        ax.grid(alpha=.18)
    axes[-1].set_xlabel('Source video time (s)')
    fig.suptitle('Landing-pad runtime: measured performance',fontsize=17,fontweight='bold',y=.98)
    fig.text(.12,.02,f'Source: {source_name} | Missing frames remain gaps; availability is not accuracy.',fontsize=9,color='#555555')
    fig.subplots_adjust(left=.14,right=.97,bottom=.085,top=.92,hspace=.42)
    fig.savefig(output/'00_overview.png',dpi=180)
    fig.savefig(output/'00_overview.svg')
    plt.close(fig)
    summary = dict(source_run=str(run_dir),processed_frames=len(rows),
                   unprocessed_frames_within_plotted_span=int(missing.sum()),
                   pose_valid_frames=count,pose_limit_working_px=limit,
                   median_processing_ms=float(np.nanmedian(processing)),
                   median_latency_ms=float(np.nanmedian(latency)),
                   median_accepted_rms_working_px=float(np.median(rms[accepted])) if accepted.any() else None)
    (output/'statistics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (output/'README.md').write_text(f'''# גרפי ביצועים להגשה

מקור הנתונים: `{run_dir}/metrics.csv`, עם סף הקבלה מתוך `results.json`.
הגרפים נוצרו מנתוני הריצה הקיימת; לא בוצעה הרצה חדשה או שינוי באלגוריתם.

## 1. זמן עיבוד והשהיה — 01_processing_time.png

הקו הכחול מציג את זמן העיבוד בתהליך המהיר, והכתום את הזמן מרגע זמינות הפריים
לאחר הפענוח והתזמון ועד לקבלת תוצאת העיבוד. ההשהיה כוללת הקטנה, העברה בתורים
ועיבוד, אך אינה כוללת פענוח וידאו או הצגה בחלון.
חציון זמן העיבוד: {summary['median_processing_ms']:.2f} מילישניות;
חציון ההשהיה: {summary['median_latency_ms']:.2f} מילישניות.
הקפיצות מצביעות על עומס משתנה; הגרף לבדו אינו מבודד את הגורם לכל קפיצה.
אין להסיק FPS כולל רק מהזמן של התהליך המהיר.

## 2. שגיאת הקרנה — 02_reprojection_error.png

הגרף מציג Reprojection RMS ואת סף קבלת ה־Pose: {limit:.4f} פיקסלים בתמונת
העבודה. בריצה זו רוחב המקור הוא {original_width} ורוחב תמונת העבודה הוא
{working_width} פיקסלים. הסף מותאם להקטנת התמונה. נקודות כתומות מציינות אומדני Pose שנפסלו.
קטעים ללא אומדן נשארים ריקים ולא מוצגים כשגיאה אפס.
שגיאה נמוכה מעידה על התאמה גיאומטרית לפינות, ולא מאמתת מרחק מול מדידה פיזית.

## 3. זמינות Pose — 03_pose_availability.png

הערך Available מציין שהתקבל Pose שעבר את בדיקות המערכת; Unavailable מציין
שלא התקבל אומדן תקין. סימונים אפורים מציינים פריימי מקור שלא עובדו.
התקבלו {count} אומדנים תקינים מתוך {len(rows)} פריימים שעובדו.
בחלק מהסרטון הדף יצא מהתמונה, ולכן ירידה בזמינות אינה בהכרח כישלון זיהוי.
זה אינו גרף של אחוז דיוק מול אמת ידועה.

## קבצים ושימוש

`00_overview.png` מאגד את שלושת הגרפים. כל גרף נשמר גם כ־SVG להגדלה ללא
אובדן חדות. הכיתובים בתוך הגרפים באנגלית; ההסברים כאן בעברית מוכנים להתאמה
למסמך ההגשה. פריימים שלא עובדו נשארים פערים בנתונים, ללא השלמת ערכים.

ליצירה חוזרת:

```powershell
python -m metrics.export_report_figures --run {run_dir}
```
''',encoding='utf-8')
    print(json.dumps(summary,indent=2))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    export(args.run,args.output)
