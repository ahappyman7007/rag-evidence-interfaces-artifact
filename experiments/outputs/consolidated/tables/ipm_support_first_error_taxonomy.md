# Support-First Error Taxonomy

Cases are paired by validation `source_id`. A case is counted when
`raw_context` has F1 at or below the raw-fail threshold and
`raw_support_first` has F1 at or above the support-success threshold.
Tags are deterministic diagnostics for sampling and writing; they are
not a substitute for final human annotation.

## Summary

|Dataset|Paired n|Raw F1|Support-first F1|Win cases|Win rate|
|---|---:|---:|---:|---:|---:|
|HotpotQA|300|0.780|0.806|7|0.023|
|2WikiMultiHopQA|300|0.782|0.822|17|0.057|
|MuSiQue|300|0.581|0.651|18|0.060|

## Primary Tags

### Grouped Tags

|Dataset|Layout bottleneck|Entity or hop confusion|Answer form near miss|Other wrong answer|
|---|---:|---:|---:|---:|
|2WikiMultiHopQA|10|5|0|2|
|HotpotQA|4|1|0|2|
|MuSiQue|14|4|0|0|

### Fine Tags

### 2WikiMultiHopQA

|Tag|Count|
|---|---:|
|some support very late|5|
|all support late|4|
|support-entity answer|4|
|other wrong answer|2|
|support split by distractors|1|
|distractor-entity answer|1|

### HotpotQA

|Tag|Count|
|---|---:|
|some support very late|2|
|other wrong answer|2|
|support split by distractors|1|
|support-entity answer|1|
|all support late|1|

### MuSiQue

|Tag|Count|
|---|---:|
|some support very late|6|
|support split by distractors|5|
|all support late|3|
|intermediate-hop answer|2|
|distractor-entity answer|2|

## Review Examples

### 2WikiMultiHopQA

1. **other wrong answer** (`18ef81360bde11eba7f7acde48001122`)
   - Q: Where was the director of film A Winter Of Cyclists born?
   - Gold: Irish
   - Raw: Munster (F1=0.0)
   - Support-first: Irish (F1=1.0)
   - Support positions: [4, 6]; titles: Mike Prendergast; A Winter of Cyclists

2. **other wrong answer** (`410ea9b20bde11eba7f7acde48001122`)
   - Q: Where was the place of burial of Princess Barbara Of Prussia's father?
   - Gold: Esparza
   - Raw: Kiel (F1=0.0)
   - Support-first: Esparza (F1=1.0)
   - Support positions: [1, 3]; titles: Prince Sigismund of Prussia (1896–1978); Princess Barbara of Prussia

3. **all support late** (`42f793240bde11eba7f7acde48001122`)
   - Q: What nationality is Amytis Of Media's husband?
   - Gold: Babylon
   - Raw: Babylonian (F1=0.0)
   - Support-first: Babylon (F1=1.0)
   - Support positions: [5, 6]; titles: Nebuchadnezzar II; Amytis of Media

4. **support-entity answer** (`52bd3f41097011ebbdb0ac1f6bf848b6`)
   - Q: Which film came out earlier, Moscow Chill or Khote Sikkey?
   - Gold: Khote Sikkey
   - Raw: Moscow Chill (F1=0.0)
   - Support-first: Khote Sikkey (F1=1.0)
   - Support positions: [0, 7]; titles: Khote Sikkey; Moscow Chill

5. **support-entity answer** (`61276448089411ebbd75ac1f6bf848b6`)
   - Q: Which film has the director died later, Taming Sutton'S Gal or Struggle For Eagle Peak?
   - Gold: Taming Sutton'S Gal
   - Raw: Struggle For Eagle Peak (F1=0.0)
   - Support-first: Taming Sutton'S Gal (F1=1.0)
   - Support positions: [1, 3, 4, 5]; titles: Struggle for Eagle Peak; Tancred Ibsen; Lesley Selander; Taming Sutton's Gal

6. **all support late** (`79f69dd80bda11eba7f7acde48001122`)
   - Q: Where was the father of Deneys Reitz born?
   - Gold: Swellendam
   - Raw: Cape Town (F1=0.0)
   - Support-first: Swellendam (F1=1.0)
   - Support positions: [6, 7]; titles: Deneys Reitz; Francis William Reitz

7. **some support very late** (`7ca3e921086811ebbd5eac1f6bf848b6`)
   - Q: Do both films, The Big Attraction and All Women Have Secrets, have the directors who are from the same country?
   - Gold: yes
   - Raw: no (F1=0.0)
   - Support-first: yes (F1=1.0)
   - Support positions: [2, 5, 6, 8]; titles: The Big Attraction; All Women Have Secrets; Max Reichmann; Kurt Neumann (director)

8. **support split by distractors** (`907764820baf11ebab90acde48001122`)
   - Q: Who is Alexander Stewart, 4Th High Steward Of Scotland's paternal grandfather?
   - Gold: Alan fitz Walter
   - Raw: Gille Brigte of Angus (F1=0.0)
   - Support-first: Alan fitz Walter (F1=1.0)
   - Support positions: [2, 5]; titles: Alexander Stewart, 4th High Steward of Scotland; Walter Stewart, 3rd High Steward of Scotland

### HotpotQA

1. **support split by distractors** (`5a79395455429970f5fffe7d`)
   - Q: What was the name of the lead character in the 1960s sitcom "Get Smart", which also featured an American actress born in 1933?
   - Gold: Maxwell Smart
   - Raw: Agent 99 (F1=0.0)
   - Support-first: Maxwell Smart (F1=1.0)
   - Support positions: [3, 6]; titles: Get Smart, Again!; Barbara Feldon

2. **some support very late** (`5a81e8ea5542995ce29dcc73`)
   - Q: Who played the role of Nettie Harris in the 1985 film directed by Steven Spielberg?
   - Gold: Akosua Gyamama Busia
   - Raw: Whoopi Goldberg (F1=0.0)
   - Support-first: Akosua Gyamama Busia (F1=1.0)
   - Support positions: [2, 8]; titles: Akosua Busia; The Color Purple (film)

3. **other wrong answer** (`5ac0d9a35542992a796ded90`)
   - Q: Are Hungry Hungry Hippos and Parcheesi both published by Parker Brothers?
   - Gold: no
   - Raw: yes (F1=0.0)
   - Support-first: no (F1=1.0)
   - Support positions: [3, 5]; titles: Hungry Hungry Hippos; Parcheesi

4. **support-entity answer** (`5ac2da27554299657fa2909f`)
   - Q: Which town was home to a forward for the Western New York Flash?
   - Gold: Oyster Bay
   - Raw: Massapequa, New York (F1=0.0)
   - Support-first: Oyster Bay (F1=1.0)
   - Support positions: [0, 7]; titles: Vicki DiMartino; Massapequa, New York

5. **all support late** (`5adf85e05542993344016cba`)
   - Q: What football league did John Moncur belong to during his time at Ipswich Town F.C.?
   - Gold: Championship
   - Raw: Premier League (F1=0.0)
   - Support-first: Championship (F1=1.0)
   - Support positions: [6, 9]; titles: John Moncur; Ipswich Town F.C.

6. **other wrong answer** (`5ae3fd2c5542995dadf2428f`)
   - Q: Are Ian Brown and Dee Snider both actors?
   - Gold: no
   - Raw: yes (F1=0.0)
   - Support-first: no (F1=1.0)
   - Support positions: [3, 5]; titles: Ian Brown; Dee Snider

7. **some support very late** (`5a734acf5542991f9a20c6ec`)
   - Q: Which singer is younger, Shirley Manson or Jim Kerr?
   - Gold: Shirley Ann Manson
   - Raw: James Kerr (F1=0.0)
   - Support-first: Shirley Manson (F1=0.8)
   - Support positions: [4, 8]; titles: Shirley Manson; Jim Kerr

### MuSiQue

1. **intermediate-hop answer** (`2hop__10114_599630`)
   - Q: Where in Zhejiang is the city where Protestants are especially notable?
   - Gold: Yongjia County
   - Raw: near Wenzhou (F1=0.0)
   - Support-first: Yongjia County (F1=1.0)
   - Support positions: [9, 12]; titles: Zhejiang; Sanjiang Church

2. **support split by distractors** (`2hop__130984_55721`)
   - Q: What place gets the most rain where Sandy High School is?
   - Gold: the Coast Range
   - Raw: coastal mountains (F1=0.0)
   - Support-first: Coast Range (F1=1.0)
   - Support positions: [7, 14]; titles: Climate of Oregon; Sandy High School

3. **all support late** (`2hop__19007_60935`)
   - Q: Who was the first person to do a full translation of the Bible into the script in which Hokkien is sometimes written?
   - Gold: St Jerome
   - Raw: Miles Coverdale (F1=0.0)
   - Support-first: St Jerome (F1=1.0)
   - Support positions: [18, 19]; titles: Vulgate; Hokkien

4. **all support late** (`2hop__242938_18803`)
   - Q: What did M. King Hubbert's employer announce it was in the process of doing in April 2010?
   - Gold: trying to find a potential buyer for all of its operations in Finland
   - Raw: divest from downstream business (F1=0.0)
   - Support-first: trying to find a potential buyer for all of its operations in Finland (F1=1.0)
   - Support positions: [10, 19]; titles: M. King Hubbert; Royal Dutch Shell

5. **support split by distractors** (`2hop__66167_88526`)
   - Q: Who was picked before the player who has the most points in an NBA season in the NBA draft?
   - Gold: Greg Oden
   - Raw: Markelle Fultz (F1=0.0)
   - Support-first: Greg Oden (F1=1.0)
   - Support positions: [1, 7]; titles: List of National Basketball Association annual scoring leaders; 2007 NBA draft

6. **some support very late** (`2hop__668407_683671`)
   - Q: Who is the father of Empress Wang's husband?
   - Gold: Yang Xingmi
   - Raw: Sima Daozi (F1=0.0)
   - Support-first: Yang Xingmi (F1=1.0)
   - Support positions: [3, 18]; titles: Empress Dowager Wang (Rui); Empress Wang (Yang Pu)

7. **intermediate-hop answer** (`2hop__725233_150107`)
   - Q: Who published Communications of the publisher of the Mobile Computing and Communications Review?
   - Gold: Association for Computing Machinery
   - Raw: ACM (F1=0.0)
   - Support-first: Association for Computing Machinery (F1=1.0)
   - Support positions: [8, 19]; titles: Mobile Computing and Communications Review; Communications of the ACM

8. **some support very late** (`2hop__78756_198548`)
   - Q: Who is the spouse of the person who voices Smokey the bear?
   - Gold: Katharine Ross
   - Raw: Laura Lizer Sommers (F1=0.0)
   - Support-first: Katharine Ross (F1=1.0)
   - Support positions: [5, 16]; titles: Smokey Bear; Murder in Texas
