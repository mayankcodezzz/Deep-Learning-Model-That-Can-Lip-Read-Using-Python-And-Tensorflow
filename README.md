# Deep-Learning-Model-That-Can-Lip-Read-Using-Python-And-Tensorflow


This project leverages deep learning to perform video-to-text transcription using a Connectionist Temporal Classification (CTC) model. The model is trained to recognize speech or actions from video frames and output a corresponding text transcription. The process begins by preprocessing video data into frame sequences, which are then passed into a recurrent neural network (RNN) model designed for sequence-to-sequence tasks. 

The project uses TensorFlow and Keras for model implementation, with the CTC loss function employed to handle the alignment between input frames and output transcriptions. A custom callback function is integrated to visualize predictions during training and to save model weights. The model is optimized using the Adam optimizer with a learning rate scheduler to adjust the learning rate as training progresses. 

Once trained, the model can process unseen video samples and predict corresponding transcriptions using CTC decoding. This approach is robust to varying sequence lengths and allows the model to predict text from videos where the alignment between input and output is not explicitly available. The model is evaluated on multiple video samples, showing its capability to transcribe speech or actions with notable accuracy, offering potential applications in automated captioning, video indexing, and accessibility tools.
# Deep LearningModelThat Can LipRead

# Using Python And Tensorflow

```
MayankPareek^1 ,TulsiPatel^2
CollegeOfScienceandHealth,DepaulUniversity,Chicago-USA
{tpatel91,mpareek}@depaul.edu
Herewearegoingtoimplementthepaper
LIPNET:END-TO-ENDSENTENCE-LEVELLIPREADING
```
Lipreading is the task of decoding text from the movement of a speaker’s
mouth.Traditionalapproachesseparatedtheproblemintotwostages:designing
or learning visual features, and prediction. More recent deep lip reading
approaches are end-to-endtrainable (Wandet al., 2016;Chung&Zisserman,
2016a). However, existing work on models trained end-to-end perform only
wordclassification,ratherthansentence-levelsequenceprediction.Studieshave
shownthathumanlipreadingperformanceincreasesforlongerwords(Easton&
Basala,1982),indicatingtheimportanceoffeaturescapturingtemporalcontext
in an ambiguous communication channel.Motivated bythis observation, we
presentLipNet,amodelthatmapsavariable-lengthsequenceofvideoframesto
text, making useofspatiotemporalconvolutions,arecurrent network,andthe
connectionist temporal classification loss, trained entirely end-to-end. Tothe
bestofourknowledge,LipNetis thefirstend-to-endsentence-levellipreading
modelthatsimultaneouslylearnsspatiotemporalvisualfeaturesandasequence
model.OntheGRIDcorpus,LipNetachieves95.2%accuracyinsentence-level,
overlappedspeakersplittask,outperformingexperiencedhumanlipreadersand
theprevious86.4%word-levelstate-of-the-artaccuracy(Gergenetal.,2016).


**ImportDependencies**
_opencv-python:Forvideoprocessing.
matplotlib:Forvisualizations.
imageio:Forsavinganimations.
gdown:FordownloadingfilesfromGoogleDrive.
tensorflow:Fordeeplearning._
**import os
import cv
import tensorflow as tf
import numpy as np
from typing import List
from matplotlib import pyplot as plt
import imageio**

ChecksforGPUavailabilityandconfiguresTensorFlowtouseGPUmemory
efficiently,avoidingmemoryallocationissues.

```
Check for GPUs Allotted:
tf.config.list_physical_devices('GPU')
physical_devices = tf.config.list_physical_devices('GPU')
try:
tf.config.experimental.set_memory_growth(physical_devices[ 0 ],
True)
except:
pass
```
NowoncewesetupeverythingwebeingwithBuildingDataLoadingFunctions
Hereweneedtwodataloadingfunctions:1sttoreadourvideosandsecondisto
pre-processourannotationswhichinthiscaseisgoingtobesomethingthatthe
personspeaks.
Hereweuse _gdown_ thatwillhelpusdownloaddatafromgoogledrive,now
herewemakeitstraightforwardbyusingthedatathatwasmeanttobeusedfor
lipreadingmodels.Nowherethedatasetisgoingtobeanextractoriwould
saysmallportionofgriddata.


```
url = 'https://drive.google.com/uc?id=1YlvpDLix3S-U8fd-gqRwPcWXAXm8JwjL'
output = 'data.zip'
gdown.download(url, output, quiet=False)
gdown.extractall('data.zip')
```
```
Herewejustsimplydownloadedthedatasetintoazipfile.
Aftercarefullyreviewingthedataweseeadatafoldercontainingalignments
ands1.Furtheralignments\s1willhave.alignfilescontainingtext
```
023750 sil
2375029500 bin
2950034000 blue
3400035500 at
3550041000 f
4100047250 two
4725053000 now
5300074500 sil
Herethisshowstheannotationsforvideobbaf2n,goingbacktodata/s
Wegetvideofileslookingatbbaf2n.mpgwegetthesameannotationspokenby
thespeakerinthevideo


Nowwehaveusedafunctiontoloadthisdata
def load_video(path:str) ->List[float]:
cap = cv2.VideoCapture(path)
frames = []
for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
ret, frame = cap.read()
frame = tf.image.rgb_to_grayscale(frame)
frames.append(frame[190:236,80:220,:])
cap.release()
mean = tf.math.reduce_mean(frames)
std = tf.math.reduce_std(tf.cast(frames, tf.float32))
return tf.cast((frames - mean),tf.float32) / std

ReadsVideoFiles,ConvertsThemToGrayscale,CropsFrames,Normalizes
PixelValues,AndReturnsPreprocessedFrames.
Here

```
frames.append(frame[190:236,80:220,:])
```

Itwillbasicallyisolatethemouthortheregionofthelip.
Nowintheoriginalpapertheyhaveusedanadvancedversiontodetectthelip
usingDLib.sohereweareloopingthroughthevideostoringtheframesinside
ourownsetofarraysframes= []Convertingitfromrgbtograyscale(Making
lessdatatopreprocess).we'rethenstandardizingitsowe'recalculatingthe
meancalculatingthestandarddeviationthat'sjustgoodpracticetoscaleyour
dataandthenwe'recastingittoafloat 32 anddividingitbythestandard
deviationsoifwegoandrunthatthatisourloadvideofunctionnowdonenow
we'regoingtogoonaheadandDefineourvocabnowavocabisreallyjust
goingtobeeverysinglecharacterwhichwemightexpecttoencounterwithin
ourannotationsobinblewatf2nowwe'vealsogotacoupleofnumbersin
thereaswell.

**FunctionDefinition**

```
def load_video(path: str) -> List[float]:
```
```
● def :Thiskeywordisusedtodefineafunction.
● load_video :Thenameofthefunction.Itindicatesthepurposeofthe
function,whichistoloadandprocessvideodata.
● path:str :Thefunctionexpectsasingleargumentpath,whichisastring
representingthefilepathtothevideo.
● ->List[float] :Thisisatypehintthatindicatesthefunctionwillreturna
listcontainingfloating-pointnumbers.
```
**OpentheVideoFile**

```
cap = cv2.VideoCapture(path)
```
```
● cv2.VideoCapture(path) :Thiscreatesavideocaptureobjectcapusing
OpenCV.Itisusedtoreadframesfromthevideofilespecifiedbythe
path.
● cap :Storesthevideocaptureobject,whichallowsaccesstovideo
propertiesandframes.
```

**InitializeanEmptyListforFrames**

```
frames = []
```
```
● frames=[] :Initializesanemptylisttostoreindividualframesfromthe
videoafterprocessing.
```
**IterateThroughVideoFrames**

```
for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
```
```
● for_ :Alooptoiterateoverarangeofnumbers.Theunderscore(_)is
usedwhentheloopvariableisn’tneeded.
● range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) :
○ cap.get(cv2.CAP_PROP_FRAME_COUNT) :Getsthetotal
numberofframesinthevideousingOpenCV.
○ int() :Convertstheframecount(afloat)intoaninteger.
○ range() :Createsarangeobjecttoloopovereachframe.
```
**ReadEachFrame**

```
ret, frame = cap.read()
```
```
● cap.read() :Readsthenextframefromthevideo.
● ret :Abooleanindicatingiftheframewasreadsuccessfully.
● frame :TheactualframedataintheformofaNumPyarray.
```
**ConvertFrametoGrayscale**

```
frame = tf.image.rgb_to_grayscale(frame)
```
```
● tf.image.rgb_to_grayscale(frame) :UsesTensorFlowtoconvertthe
framefromRGB(color)tograyscale.Thegrayscaleframehasone
channelinsteadofthree.
```

**CropFrame**

```
frames.append(frame[190:236, 80:220, :])
```
```
● frame[190:236,80:220,:] :Cropstheframetoaspecificregion.
○ 190:236 :Selectsrows(height)from 190 to 236 (inclusive).
○ 80:220 :Selectscolumns(width)from 80 to 220 (inclusive).
○ : :Keepsallgrayscalechannels(thoughgrayscalehasonlyone
channel).
● frames.append(...) :Addsthecroppedframetotheframeslist.
```
**ReleaseVideoCapture**

```
cap.release()
```
```
● cap.release() :Freestheresourcesassociatedwiththevideocapture
object.
```
**CalculateMeanandStandardDeviation**

```
mean = tf.math.reduce_mean(frames)
std = tf.math.reduce_std(tf.cast(frames, tf.float32))
```
```
● tf.math.reduce_mean(frames) :Computesthemeanvalueofallpixel
intensitiesacrossallframes.
● tf.math.reduce_std(...) :Computesthestandarddeviationofallpixel
intensities.Itrequirestheinputtobecasttoafloatdatatype.
○ tf.cast(frames,tf.float32) :Convertsthedatatypeofframesfrom
itsoriginaltype(likelyuint8)tofloat32fornumericalstability.
```

**NormalizetheFrames**

```
return tf.cast((frames - mean), tf.float32) / std
```
```
● frames-mean :Subtractsthemeanfromeachframetocenterthepixel
valuesaroundzero.
● tf.cast(...,tf.float32) :Ensurestheresultingarrayisoftypefloat32.
● /std :Dividesbythestandarddeviationtonormalizethepixelvalues.
Afterthisoperation:
○ Pixelvalueshaveameanof0.
○ Pixelvalueshaveastandarddeviationof1.
● return :Outputsthelistofnormalizedframes.
```
**Summary**

Thefunction:

1. Readsavideofile.
2. Convertseachframetograyscale.
3. Cropseachframetoaspecificregionofinterest.
4. Normalizesthepixelintensitiesofallframesusingmeanandstandard
    deviation.
5. Returnsalistofnormalizedframesasfloat32tensors.

Nowcreatingalistvocab

```
vocab = [x for x in"abcdefghijklmnopqrstuvwxyz'?!123456789 "]
```
```
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab,oov_token="")
num_to_char = tf.keras.layers.StringLookup(
vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)
print(
f"Thevocabulary is: {char_to_num.get_vocabulary()}"
f"(size={char_to_num.vocabulary_size()})"
)
```

Thiscodesnippetispartofaprocessthatmapscharacterstonumericalindices
andviceversa,usingTensorFlow'sStringLookuplayer.Here’sadetailed
breakdownofeachline:

**Character-to-NumberMapping**

```
char_to_num =
tf.keras.layers.StringLookup(vocabulary=vocab,
oov_token="")
```
```
● char_to_num :ThisvariablestoresaninstanceoftheStringLookuplayer,
whichisusedtomapstrings(characters)touniquenumericalindices.
● tf.keras.layers.StringLookup(...) :CreatesaTensorFlowlayerthat
performsstring-to-indexconversions.Itassignsanintegerindextoeach
characterinthevocabulary.
○ vocabulary=vocab :
■ vocab :Alistorarrayofuniquecharactersusedinthe
mapping(e.g.,['a','b','c',...]).
■ Thevocabparameterspecifiesthesetofcharactersthatwill
bemappedtoindices.
○ oov_token="" :
■ Theoov_token(Out-Of-Vocabularytoken)isusedfor
charactersthatarenotfoundintheprovidedvocabulary.
■ Here,anemptystring("")isspecifiedasthe
out-of-vocabularytoken.
```
**Number-to-CharacterMapping**

```
num_to_char = tf.keras.layers.StringLookup(
vocabulary=char_to_num.get_vocabulary(),
oov_token="", invert=True
)
```

```
● num_to_char :ThisvariablestoresanotherStringLookuplayer,butitis
configuredtoperformthereverseoperation—mappingindicesbackto
characters.
● tf.keras.layers.StringLookup(...) :Createsanewinstanceofthe
StringLookuplayer.
○ vocabulary=char_to_num.get_vocabulary() :
■ char_to_num.get_vocabulary() :Retrievesthevocabulary
fromthechar_to_numlayer.Thisensuresthatthetwolayers
(char_to_numandnum_to_char)usethesamemapping.
○ oov_token="" :
■ Specifiestheout-of-vocabularytokenforthisreverse
mapping.
○ invert=True :
■ Theinvertparametertellsthelayertomapindicesbackto
stringsinsteadofstringstoindices.
```
**PrinttheVocabularyandItsSize**

```
print(
f"The vocabulary is: {char_to_num.get_vocabulary()} "
f"(size ={char_to_num.vocabulary_size()})"
)
```
```
● print(...) :Outputsinformationaboutthevocabularyusedbythe
char_to_numlayer.
○ f"Thevocabularyis:{char_to_num.get_vocabulary()}" :
■ f"..." :Thisisanf-string,whichallowsembeddingvariables
andexpressionsinsidecurlybraces{}fordynamicstring
formatting.
■ char_to_num.get_vocabulary() :
■ Retrievesthelistofalluniquecharactersinthe
vocabulary.Itincludesallcharactersfromthevocab
listprovidedearlier,alongwithanyspecialtokenslike
theout-of-vocabularytoken(ifapplicable).
■ Thispartofthef-stringdisplaysthelistofcharactersinthe
vocabulary.
```

```
○ f"(size={char_to_num.vocabulary_size()})" :
■ char_to_num.vocabulary_size() :
■ Retrievesthetotalsizeofthevocabulary,which
includesalluniquecharactersandpossiblythe
out-of-vocabularytoken.
■ Thispartofthef-stringdisplaysthesizeofthevocabulary.
```
**ExampleWalkthrough**

Let’sassumethefollowing:

```
● vocab=['a','b','c','d']
```
**Step-by-StepProcess:**

1. **Definechar_to_num** :
    ○ Maps'a'->1,'b'->2,'c'->3,'d'-> 4 (index 0 isreservedfor
       padding).
    ○ Ifacharacterisnotinthevocabulary(e.g.,'z'),itismappedtothe
       out-of-vocabularytoken"".
2. **Definenum_to_char** :
    ○ Reversesthemapping:
       ■ 1 ->'a', 2 ->'b', 3 ->'c', 4 ->'d'.
3. **PrinttheVocabulary** :
    ○ char_to_num.get_vocabulary()outputs['[UNK]','a','b','c','d'],
       where[UNK]representstheunknowntoken.
    ○ char_to_num.vocabulary_size()outputs5,indicatingthetotal
       numberoftokens(including[UNK]).

**Output**

Ifvocab=['a','b','c','d'],theoutputoftheprintstatementwillbe:

Thevocabularyis:['[UNK]','a','b','c','d'](size=5)


```
def load_alignments(path:str) -> List[str]:
with open(path,'r') asf:
lines = f.readlines()
tokens = []
for line inlines:
line = line.split()
if line[ 2 ] !='sil':
tokens = [*tokens,' ',line[ 2 ]]
return char_to_num(tf.reshape(tf.strings.unicode_split(tokens,
input_encoding='UTF-8'), (-1)))[ 1 :]
```
Thisfunction,load_alignments,processesatextfilecontainingalignment
information(presumablyphoneticorlinguisticdata)andconvertsitinto
numericaltokensusingthepreviouslydefinedchar_to_nummapping.Hereisa
detailedexplanationofthecode,linebyline:

**FunctionDefinition**

```
def load_alignments(path: str) -> List[str]:
```
```
● defload_alignments :Declaresafunctionnamedload_alignments.
● path:str :Theparameterpathexpectsastringrepresentingthepathtothe
alignmentfile.
● ->List[str] :Indicatesthatthefunctionreturnsalistofstrings.
```
**OpentheFileandReadLines**

```
with open(path, 'r') as f:
lines = f.readlines()
```
```
● withopen(path,'r')asf: :
○ Opensthefileatthespecifiedpathinreadmode('r').
○ Thewithstatementensuresthatthefileisautomaticallyclosed
whentheblockisexited.
○ f :Afileobjectrepresentingtheopenedfile.
● lines=f.readlines() :
○ Readsalllinesofthefileintoalistcalledlines.
○ Eachlineinthefilebecomesanelementinthislistasastring.
```

**InitializeTokensList**

```
tokens = []
```
```
● Initializesanemptylistcalledtokens.Thiswillstoreprocessedtokens
extractedfromthefile.
```
**IterateThroughLines**

```
for line in lines:
line = line.split()
if line[ 2 ] != 'sil':
tokens = [*tokens, ' ', line[ 2 ]]
```
**Line-by-LineBreakdown:**

1. **forlineinlines:** :
    ○ Iteratesovereachlineinthelineslist.
    ○ **line** :Representsthecurrentlinebeingprocessed.
2. **line=line.split()** :
    ○ Splitsthecurrentlineintoalistofwordsorelements,using
       whitespaceasthedelimiter.
    ○ **line** nowcontainsthelineasalistofitsindividualcomponents.
3. **ifline[2]!='sil':** :
    ○ Checkswhetherthethirdelement(line[2])inthesplitlineisnot
       equaltothestring'sil'.
    ○ **'sil'** representssilence,whichisskipped.
4. **tokens=[*tokens,'',line[2]]** :
    ○ Usesunpacking(*tokens)toappendaspace('')followedbythe
       thirdelement(line[2])tothetokenslist.
    ○ **Purpose** :Thisensuresproperspacingbetweentokenswhile
       buildingthesequence.


**ConvertTokenstoNumericForm**

```
return char_to_num(
tf.reshape(
tf.strings.unicode_split(tokens,
input_encoding='UTF-8'),
(- 1 )
)
)[ 1 :]
```
**Step-by-StepExplanation:**

1. **tf.strings.unicode_split(tokens,input_encoding='UTF-8')** :
    ○ SplitseachstringinthetokenslistintoitsindividualUnicode
       characters.
    ○ **tokens** :Containsasequenceofstrings(e.g.,['','a','b','']).
    ○ **input_encoding='UTF-8'** :Specifiestheencodingoftheinput
       strings.
2. **tf.reshape(...,(-1))** :
    ○ Flattenstheresultingtensorintoa1Dtensorwithshape(-1)(all
       elementsinasingledimension).
    ○ Thisisneededforcompatibilitywithchar_to_num.
3. **char_to_num(...)** :
    ○ ConvertsthecharactersfromtheUnicode-splittensorinto
       numericalindicesusingthechar_to_nummappinglayer.
4. **[1:]** :
    ○ Removesthefirstelementofthenumericalsequence.
    ○ Thismayexcludealeadingpaddingorotherspecialtokenadded
       duringprocessing.

**WhattheFunctionDoes**

1. Opensatextfilecontainingalignmentinformation.
2. Extractstokensfromeachline,skippingtokenslabeledas'sil'.
3. Convertsthetokensintoaflatsequenceofnumericalindicesusingthe
    char_to_nummapping.


**ExampleWalkthrough**

Thefilecontains:

```
0 23750 sil
23750 29500 bin
29500 34000 blue
34000 35500 at
35500 41000 f
41000 47250 two
47250 53000 now
53000 74500 sil
```
**Step-by-StepExecution**

1. **ReadFileContents**

**Lines** :Afterreading,thelineslistwillbe:

```
[
"0 23750 sil",
"23750 29500 bin",
"29500 34000 blue",
"34000 35500 at",
"35500 41000 f",
"41000 47250 two",
"47250 53000 now",
"53000 74500 sil"
]
```
2. **Initializetokens** :
    ○ Anemptylisttokens=[]iscreatedtostoreprocessedtokens.
3. **ProcessEachLine**


```
○ Loopingthrougheachline,thecodesplitsthelineandchecksifthe
thirdelement(line[2])is not 'sil'.Ifit'snot'sil',itappendsaspace
('')andthethirdelement(line[2])totokens.
```
```
○ IterationDetails :
■ Line 1 ("0 23750 sil"):Skipped(line[2]=='sil').
■ Line 2 ("23750 29500 bin"):Adds''and'bin'totokens.
■ tokens=['','bin']
■ Line 3 ("29500 34000 blue"):Adds''and'blue'totokens.
■ tokens=['','bin','','blue']
■ Line 4 ("34000 35500 at"):Adds''and'at'totokens.
■ tokens=['','bin','','blue','','at']
■ Line 5 ("35500 41000 f"):Adds''and'f'totokens.
■ tokens=['','bin','','blue','','at','','f']
■ Line 6 ("41000 47250 two"):Adds''and'two'totokens.
■ tokens=['','bin','','blue','','at','','f','','two']
■ Line 7 ("47250 53000 now"):Adds''and'now'totokens.
■ tokens=['','bin','','blue','','at','','f','','two','',
'now']
■ Line 8 ("53000 74500 sil"):Skipped(line[2]=='sil').
```
FinaltokensList:

```
tokens=['','bin','','blue','','at','','f','','two','','now']
```
4. **UnicodeSplitting**
    ○ **tf.strings.unicode_split(tokens,input_encoding='UTF-8')** :

Splitsthetokenslistintoindividualcharacters,includingspaces:

```
['','b','i','n','','b','l','u','e','','a','t','','f','','t','w','o','','n','o','w']
```
5. **Reshaping**
    ○ **tf.reshape(...,(-1))** :

Flattensthesequenceintoasingle-dimensionaltensor:


```
['','b','i','n','','b','l','u','e','','a','t','','f','','t','w','o','','n','o','w']
```
6. **ConvertCharacterstoNumericTokens**
    ○ **char_to_num(...)** :
       ■ Mapseachcharactertoacorrespondingnumericalindex
          usingthechar_to_numlayer.

Examplemapping(assumingavocabularylike['','a','b',...,'z']):

```
' ' -> 0
'b' -> 2
'i' -> 9
'n' -> 14
'l' -> 12
'u' -> 21
...
```
ResultingNumericSequence:

```
[ 0 , 2 , 9 , 14 , 0 , 2 , 12 , 21 , 5 , 0 , 1 , 20 , 0 , 6 , 0 , 20 , 23 , 15 , 0 , 14 , 15 , 23 ]
```
7. **RemoveLeadingElement**
    ○ **[1:]** :
       ■ Removesthefirstelement(oftenapaddingorspacetoken).

FinalOutput:

```
[ 2 , 9 , 14 , 0 , 2 , 12 , 21 , 5 , 0 , 1 , 20 , 0 , 6 , 0 , 20 , 23 , 15 , 0 , 14 , 15 , 23 ]
```
**Summary**

Thefunctionprocessesthealignmentfileto:

1. Excludelineswith'sil'.
2. Createasequenceofcharacters(withspacesbetweenwords).
3. Convertcharacterstonumericalindicesusingthechar_to_nummapping.


4. Returntheprocessedsequenceasatensorofnumerictokens,excluding
    theleadingelement.

Foryourexample,thesilencetokens('sil')atthestartandendareignored,and
thefunctionoutputsthenumericalindicesrepresentingthewordsbinblueatf
twonow.

```
def load_data(path: str):
path = bytes.decode(path.numpy())
#file_name = path.split('/')[-1].split('.')[ 0 ]
# File namesplitting for windows
file_name = path.split('\\')[-1].split('.')[ 0 ]
video_path = os.path.join('data','s1',f'{file_name}.mpg')
alignment_path = os.path.join('data','alignments','s1',f'{file_name}.align')
frames = load_video(video_path)
alignments = load_alignments(alignment_path)
return frames, alignments
```
Let'sanalyzetheload_datafunctionindetailstepbystep,coveringits
functionalityandroleinloadingvideoandalignmentdata:

**CodeAnalysis**

**1.DecodePath(StringConversion)**

```
path = bytes.decode(path.numpy())
```
```
● Input :ThepathparameterisinitiallyaTensorFlowtensor(likelyoftype
tf.stringcontainingthepathtoafile).
● Purpose :
○ path.numpy()convertstheTensorFlowtensorintoaNumPybyte
string.
○ bytes.decode(...)thenconvertsthebytestringintoaregularPython
string.
● Result :pathisnowaplainPythonstringrepresentingthefilepath.
```

**2.ExtractFileName**

```
file_name = path.split('\\')[- 1 ].split('.')[ 0 ]
```
```
● Input :pathisastringrepresentingthefullfilepath,e.g.,
C:\data\video001.align.
● Purpose :
○ SplitsthepathbytheWindowsdirectoryseparator(\\).
○ Takesthelastelementofthesplitresult(e.g.,video001.align).
○ Splitsthatbythefileextensionseparator(.)andselectsthefirst
part(e.g.,video001).
● Result :file_namecontainsthebasefilenamewithouttheextension
(video001inthisexample).
```
**3.ConstructVideoPath**

```
video_path = os.path.join('data', 's1', f'{file_name}.mpg')
```
```
● Input :Thefile_namederivedearlier(video001).
● Purpose :
○ Constructsthepathtothecorrespondingvideofile.
○ Assumesthevideosarestoredinadirectorystructurestartingwith
'data/s1'andarenamed{file_name}.mpg.
```
**Result** :Forfile_name="video001",video_pathwouldbe:

```
data/s1/video001.mpg
```
**4.ConstructAlignmentPath**

```
alignment_path = os.path.join('data', 'alignments', 's1', f'{file_name}.align')
```
```
● Input :Thefile_namederivedearlier(video001).
● Purpose :
○ Constructsthepathtothecorrespondingalignmentfile.
○ Assumesalignmentsarestoredin'data/alignments/s1'andare
named{file_name}.align.
```

**Result** :Forfile_name="video001",alignment_pathwouldbe:

```
data/alignments/s1/video001.align
```
**5.LoadVideoData**

```
frames = load_video(video_path)
```
```
● Purpose :
○ Callstheload_videofunctiontoreadandpreprocessframesfrom
thevideofileatvideo_path.
● Result :
○ framescontainsaprocessedrepresentationofvideoframes,suchas
anormalizedtensororalistofframes.
```
**6.LoadAlignmentData**

```
alignments = load_alignments(alignment_path)
```
```
● Purpose :
○ Callstheload_alignmentsfunctiontoreadandpreprocessthe
alignmentinformationfromthealignmentfileatalignment_path.
● Result :
○ alignmentscontainstheprocessedalignmenttokensasatensoror
list.
```
**7.ReturnLoadedData**

```
return frames, alignments
```
```
● Purpose :
○ Combinesthevideoframedata(frames)andalignmenttokendata
(alignments)intoatupleandreturnsit.
● Result :
```

```
○ Thefunctionreturnsatuple(frames,alignments)thatcanbeused
infurtherprocessing,suchasmodeltrainingorinference.
```
**Summary**

Theload_datafunction:

1. DecodestheTensorFlowtensorpathintoaPythonstring.
2. Extractsthebasefilenamefromtheinputpath.
3. Constructspathsforthecorrespondingvideoandalignmentfilesbasedon
    apredefineddirectorystructure.
4. Loadsandpreprocessesvideoframesusingload_video.
5. Loadsandpreprocessesalignmentdatausingload_alignments.
6. Returnsthepreprocessedframesandalignmentsasatuple.

```
test_path = '.\\data\\s1\\bbal6n.mpg'
tf.convert_to_tensor(test_path).numpy().decode('utf-8').split('\\')[- 1 ].split('.
')[ 0 ]
frames, alignments = load_data(tf.convert_to_tensor(test_path))
plt.imshow(frames[ 40 ])
```
Thisfunctionservesasabridgetoconnectrawfilepathswiththeirprocessed
contents,ensuringthedataisinthecorrectformatfordownstreamtasks.


**CodeAnalysis**

```
frames, alignments = load_data(tf.convert_to_tensor(test_path))
```
**1.tf.convert_to_tensor(test_path)**

```
● Purpose :ConvertsthePythonvariabletest_path(whichisexpectedtobe
astringrepresentingafilepath)intoaTensorFlowtensoroftypetf.string.
● Input :
○ test_pathisastring,e.g.,"data/test_video.align".
● WhyTensor?
○ TensorFlowoftenrequirestensorsasinputs,especiallywhenusing
itsfunctionsordatasetspipeline.
● Output :ATensorFlowtensorrepresentationofthefilepath.
```
Example:

```
test_path = "data/test_video.align"
tf.convert_to_tensor(test_path)
```
```
# Output: < tf.Tensor: shape=(), dtype=string,
numpy=b'data/test_video.align'>
```
**2.load_data(...)**

```
● Purpose :Callstheload_datafunctionwiththetensorizedfilepath.
● Behavior :
○ Internally,load_datadecodesthetensortoastringpath
(bytes.decode(path.numpy())).
○ Itthenprocessesthevideoandalignmentfileslocatedatthepaths
derivedfromtest_path.
```

**3.frames,alignments=...**

```
● Purpose :Unpacksthereturnvalueofload_dataintotwovariables:
○ frames:Representstheprocessedvideoframesloadedand
normalizedbytheload_videofunction.
○ alignments:Representsthealignmenttokensloadedand
transformedbytheload_alignmentsfunction.
```
**ExampleWorkflow**

Assume:

```
● test_path="data/s1/video001.align"
```
Execution:

1. test_pathisconvertedintoatensor:
    tf.convert_to_tensor("data/s1/video001.align").
2. load_dataprocessesthevideo(data/s1/video001.mpg)andalignment
    (data/alignments/s1/video001.align)files:
       ○ **Video** :Normalizestheframesandextractsaspecificregion.
       ○ **Alignment** :Convertstexttokensintonumericalrepresentations.
3. Theoutputsareunpacked:
    ○ framescouldbeatensororlistofpreprocessedvideoframes.
    ○ alignmentscouldbeatensororlistofnumericalalignmenttokens.

Frame0:


Frame11:

Frame25:

CoolRight?

NowYoureadsofarthankyou!

```
tf.strings.reduce_join([bytes.decode(x) for x in
num_to_char(alignments.numpy()).numpy()])
```
1. **AlignmentsTensor** :
    ○ Thealignmentstensorcontainsnumericaltokensrepresenting
       words(orsubwords)extractedfromthealignmentfile.
    ○ Forexample,ifalignments.numpy()was[2,3,4,5,6],itwould
       correspondtotokensbasedonthevocabulary.
2. **num_to_charTransformation** :
    ○ Thenum_to_charmappinglayertranslatesnumericalindicesinto
       theirstringequivalents.
    ○ Ifyourvocabularywassomethinglike['','','bin','blue','at','l',
       'six','now','sil'],then:
          ■ 2 ->'bin'
          ■ 3 ->'blue'
          ■ 4 ->'at'
          ■ 5 ->'l'
          ■ 6 ->'six'
          ■ 7 ->'now'


3. **Decoding** :
    ○ Thebytes.decode(x)operationconvertsthesebytes-likestringsinto
       regularPythonstrings.

Afterdecoding,thelistmightlooklike:

```
['bin', 'blue', 'at', 'l', 'six', 'now']
```
4. **Joining** :

Thetf.strings.reduce_join()functionconcatenatesthesestringsintoasingle
stringwithnodelimiterbydefault:

```
'binblueatlixnow'
```
Ifthereareimplicitspacesfromthealignmentormanualadjustments,the
resultwillincludethem:

```
'bin blue at l six now'
```
**WhyThisResult?**

```
● Thenumericaltokenslikelymappedtowordsvianum_to_char.
● Thesewordsinclude:
○ 'bin'
○ 'blue'
○ 'at'
○ 'l'
○ 'six'
○ 'now'
● Therewerespacesordelimitersintheoriginaldatathatwerepreserved
duringconcatenation.
```
Thefinalresult:

```
< tf.Tensor: shape=(), dtype=string, numpy=b'bin blue at l
six now'>
```

**OutcomeAnalysis**

Thestringrepresentationreflectsthealignmenttokensconvertedbacktotheir
originalwords:

```
● "binblueatlsixnow"
```
Thismatchestheprocessedalignmentdata,showingthewordsspokeninthe
videosegmentwithanysilence(sil)removed.

```
def mappable_function(path:str) ->List[str]:
result = tf.py_function(load_data, [path], (tf.float32, tf.int64))
return result
```
Themappable_functionyouprovidedisdefinedtowraparoundafunctionthat
loadsdata(likelyvideoframesandalignments)fromagivenfilepath.Ituses
tf.py_functiontoinvokeaPythonfunctionwithinTensorFlow'sgraph.

Let’sbreakdowneachpart:

**1.mappable_function(path:str)->List[str]**

```
● Thisdefinesafunctionmappable_function,whichtakesapathargument
(likelyastring)andreturnsalistofstrings.
● Thepurposeofthisfunctionistoactasawrapperfortheload_data
function.
```
**2.result=tf.py_function(load_data,[path],(tf.float32,tf.int64))**

```
● tf.py_function:ThisisaTensorFlowoperationthatallowsyoutoexecute
aPythonfunctionwithinaTensorFlowcomputationgraph.Itallowsyou
tousecustomPythoncodethatisn'tdirectlysupportedbyTensorFlow,
butstillletsyouintegrateitintoTensorFlowoperations.
○ Firstargument(load_data) :ThisisthePythonfunctiontobe
called,whichtakesinthepathandloadsthevideoframesand
alignments.Thefunctionitselfisexpectedtoreturntwothings:
■ Videoframes(likelyatensoroftypetf.float32).
■ Alignments(likelyatensoroftypetf.int64).
```

```
○ Secondargument([path]) :Thelistofargumentsthatwillbe
passedtotheload_datafunction.Here,itcontainsthepathasinput.
○ Thirdargument((tf.float32,tf.int64)) :Thisspecifiesthe
expectedoutputtypesofthefunction.Inthiscase,load_data
shouldreturntwotensors:
■ Thefirstoutput(frames)shouldbeoftypetf.float32(likely
normalizedpixelvaluesfromvideoframes).
■ Thesecondoutput(alignments)shouldbeoftypetf.int64
(indicesrepresentingwordsortokensfromthealignment).
```
**3.returnresult**

```
● Theresultreturnedfromtf.py_functionisatuplecontainingthetwo
outputsfromload_data,butwrappedinTensorFlowtensors.
● Thesetensorswillhavethespecifiedtypes(tf.float32,tf.int64),andsince
themappable_functionissupposedtoreturnalistofstrings,youlikely
needtofurtherprocesstheseoutputs,especiallyforthealignments
(convertingtf.int64toalistofstrings).
```
## CreatingDataPipeline

```
from matplotlib importpyplotas plt
```
```
data = tf.data.Dataset.list_files('./data/s1/*.mpg')
data = data.shuffle(500, reshuffle_each_iteration=False)
data = data.map(mappable_function)
data = data.padded_batch(2,padded_shapes=([75,None,None,None],[40]))
data = data.prefetch(tf.data.AUTOTUNE)
# Added for split
train = data.take(450)
test = data.skip(450)
```
**1.data=tf.data.Dataset.list_files('./data/s1/*.mpg')**

```
● Purpose :ThislinecreatesaTensorFlowDatasetbylistingall.mpgvideo
filesinthedirectory./data/s1/.
● Explanation :
○ tf.data.Dataset.list_filestakesafilepathpattern(inthiscase,
./data/s1/*.mpg)andcreatesadatasetcontainingthefilepaths
```

```
matchingthepattern.
○ Theresultingdataobjectisadatasetoffilepaths(strings)tothe
.mpgvideofiles.
```
**2.data=data.shuffle(500,reshuffle_each_iteration=False)**

```
● Purpose :Thislineshufflesthedatasettorandomizetheorderofthe
videofiles.
● Explanation :
○ shuffle(500)specifiesthatabufferof 500 elementswillbe
maintainedinmemory,andelementswillberandomlysampled
fromthisbufferforshuffling.Afterconsuminganelement,the
bufferisfilledwithanewelement.
○ reshuffle_each_iteration=Falsemeansthedatasetwillnotbe
reshuffledatthestartofeachnewepoch.IfsettoTrue,thedata
wouldbereshuffledatthebeginningofeachiteration(epoch).
```
**3.data=data.map(mappable_function)**

```
● Purpose :Thislineappliesatransformationfunction
(mappable_function)toeachelementinthedataset.
● Explanation :
○ Themapfunctionallowsyoutoapplyanycustomtransformation
tothedataset.Here,mappable_functionisappliedtoeachfilepath
inthedataset.
○ mappable_functionprocesseseachfilepath,loadingthevideo
framesandalignments.
```
**4.data=data.padded_batch(2,
padded_shapes=([75,None,None,None],[40]))**

```
● Purpose :Thislinebatchesthedataintobatchesofsize2,andpadsthe
sequencestoensuretheyhaveconsistentshapes.
● Explanation :
○ padded_batch(2)createsbatchesof 2 elements(i.e., 2 examples
fromthedataset).
○ padded_shapes=([75,None,None,None],[40])specifiesthe
shapesforpadding:
■ Forvideoframes(thefirstpartofthetuple):[75,None,
```

```
None,None]impliesthatthevideoframeswillhaveashape
of(75,height,width,channels),whereNoneindicatesa
variablesizeforheightandwidth.Thenumber 75 could
representthefixednumberofframes.
■ Foralignments(thesecondpartofthetuple):[40]specifies
thateachalignmentsequencewillhaveafixedlengthof 40
tokens.
○ Paddingwillbeappliedtosequencessothateachbatchcontains
elementsofconsistentshape.
```
**5.data=data.prefetch(tf.data.AUTOTUNE)**

```
● Purpose :Thislinesetsupprefetchingtoimproveinputpipeline
performance.
● Explanation :
○ prefetch(tf.data.AUTOTUNE)allowsTensorFlowto
asynchronouslyloadthenextbatchofdatawhilethecurrentbatch
isbeingprocessedbythemodel.Thisimprovestheoverall
throughputofthedatapipeline.
○ AUTOTUNEautomaticallydeterminestheoptimalnumberof
elementstoprefetchbasedonsystemperformance.
```
**6.train=data.take(450)**

```
● Purpose :Thislinecreatesthetrainingdatasetbytakingthefirst 450
elementsfromthedatapipeline.
● Explanation :
○ take(450)retrievesthefirst 450 batchesfromthedataset.These
batcheswillbeusedfortraining.
```
**7.test=data.skip(450)**

```
● Purpose :Thislinecreatesthetestdatasetbyskippingthefirst 450
elementsinthedataset.
● Explanation :
○ skip(450)skipsthefirst 450 batches,leavingtherestofthedataset
(afterthose 450 batches)fortesting.
○ Thisensuresthatthetrainingandtestdatasetsaresplit
appropriately.
```

**Summary**

Thispipeline:

```
1 .Loads.mpgfilesfromthe./data/s1/directory.
```
2. Shufflesthedatasetwithabuffersizeof500,withoutreshufflingbetween
    epochs.
3 .Appliesthemappable_functiontoeachfilepath,whichloadsthevideo
    framesandalignments.
4. Paddstheframesandalignmentstoconsistentshapesandbatchesthem
    intobatchesof2.
5. Prefetchesdataforperformanceoptimization.
6. Splitsthedatasetintoatrainingsetwiththefirst 450 batchesandatest
    setwiththeremainingbatches.

Thispipelineiswell-suitedfortrainingamodelonvideodatawith
correspondingalignmenttokens.

###### GOT MY ERROR

```
frames, alignments = data.as_numpy_iterator().next()
```

###### Come on errors are part of life or i would

###### say errors are life it makes you feel more

###### connected with the code that you wrote(of

###### course you wrote not gpt)

###### Let’s Debug it!

###### The issue is in how the file name is being

###### extracted. When you use split('.')[0] on a

###### filename like prii8p.mpg, it's removing the

###### entire filename, leaving an empty string.

###### OF course it didn’t took me 3 hours and its 5

###### in the morning! I sometimes hate coding but

###### in the end i love this let’s see what else is

###### about to come next!

```
def load_data(path: str):
path = bytes.decode(path.numpy())
print("Original path:", path)
# Corrected filename extraction
file_name = path.split('/')[-1].split('.')[ 0 ] #For Unix-like paths
# Alternativefor Windows: file_name =
path.split('\\')[-1].split('.')[ 0 ]
print("Extractedfile name:", file_name)
# Video path
video_path =os.path.join('data','s1',f'{file_name}.mpg')
print("Videopath:", video_path)
# Alignment path
alignment_path =
os.path.join('data','alignments','s1',f'{file_name}.align')
print("Alignmentpath:", alignment_path)
# Check iffilesactually exist
if not os.path.exists(video_path):
print(f"Videofile not found: {video_path}")
raise FileNotFoundError(f"Videofile not found: {video_path}")
```
```
if not os.path.exists(alignment_path):
```

```
print(f"Alignmentfile not found: {alignment_path}")
raise FileNotFoundError(f"Alignmentfile not found:
{alignment_path}")
frames =load_video(video_path)
alignments =load_alignments(alignment_path)
return frames, alignments
```
**Updateitandlet'smoveon
frames, alignments = data.as** **__numpy__** **iterator().next()
len(frames)
sample = data.as** **__numpy__** **iterator()
val = sample.next(); val[0]
imageio.mimsave('./animation.gif', val[ 0 ][ 0 ],fps=10)**

1.frames,alignments=data.as_numpy_iterator().next()

```
● Whatitdoes:ConvertstheTensorFlowdatasetdataintoaNumPyiterator
andretrievesthefirstbatchofdata.
○ data.as_numpy_iterator()convertsthedatasetintoagenerator-like
objectthatyieldsNumPyarrays.
○ .next()fetchesthefirstbatchofdatafromtheiterator.
○ framesandalignments:
■ frames:Abatchofvideoframes,storedasaNumPyarray.
■ alignments:Correspondingalignmentlabels(e.g.,
lip-readingtextorphonemesequences).
```
2.len(frames)

```
● Whatitdoes:Calculatesthenumberofvideoframesintheframes
variable.
○ Thisisusefultounderstandthesizeoftheinputvideobatch.
● Output:Thetotalnumberofframesinthecurrentvideo.
```
3.sample=data.as_numpy_iterator()

```
● Whatitdoes:Createsareusableiteratoroverthedataset.
○ Thesamplevariablestorestheiteratorforfetchingbatchesofdata
```

```
fromdata.
```
4.val=sample.next();val[0]

```
● Whatitdoes:
○ val=sample.next():Retrievesthenextbatchofdatafromthe
sampleiterator.
■ valisatuplecontainingtwoelements:(frames_batch,
alignments_batch).
○ val[0]:Extractstheframes_batch(videoframes)fromthecurrent
batch.
```
5.imageio.mimsave('./animation.gif',val[0][0],fps=10)

```
● Whatitdoes:
○ imageio.mimsave:SavesasequenceofimagesasananimatedGIF.
○ Parameters:
■ ./animation.gif:Pathtosavetheresultinganimation.
■ val[0][0]:Thefirstsetofframes(i.e.,thefirstvideointhe
batch).
■ fps=10:FramespersecondfortheGIFanimation.
● Purpose:Createsavisualrepresentationofthefirstvideointhebatchby
animatingitsframes.
```
Step-by-StepFlow

1. FetchBatchofData:Retrievevideoframes(frames)andalignments
    (alignments)usingdata.as_numpy_iterator().next().
2. AnalyzeFrames:Checkthenumberofframesusinglen(frames).
3. IterateThroughDataset:Usesample.next()tosequentiallyaccessbatches
    ofdata.
4. CreateAnimation:SavethefirstvideoofthebatchasananimatedGIF
    usingimageio.mimsave.


FinalOutput

```
● GIFFile:Afilenamedanimation.gifiscreated,showingthefirstvideo’s
framesat 10 framespersecond.
● PracticalUse:Thisvisualizationcanbehelpfulfordebuggingor
verifyingthatthevideopreprocessingpipelineisworkingcorrectly.
```
##### DesigntheDeepNeuralNetwork

```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout,
Bidirectional, MaxPool3D, Activation,Reshape, SpatialDropout3D,
BatchNormalization, TimeDistributed, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint,
LearningRateScheduler
data.as_numpy_iterator().next()[ 0 ][ 0 ].shape
model = Sequential()
model.add(Conv3D( 128 , 3 , input_shape=( 75 , 46 , 140 , 1 ), padding='same'))
model.add(Activation('relu'))
model.add(MaxPool3D(( 1 , 2 , 2 )))
```
```
model.add(Conv3D( 256 , 3 , padding='same'))
model.add(Activation('relu'))
model.add(MaxPool3D(( 1 , 2 , 2 )))
```
```
model.add(Conv3D( 75 , 3 , padding='same'))
model.add(Activation('relu'))
model.add(MaxPool3D(( 1 , 2 , 2 )))
```
```
model.add(TimeDistributed(Flatten()))
```
```
model.add(Bidirectional(LSTM( 128 , kernel_initializer='Orthogonal',
return_sequences=True)))
model.add(Dropout(.5))
```
```
model.add(Bidirectional(LSTM( 128 , kernel_initializer='Orthogonal',
return_sequences=True)))
model.add(Dropout(.5))
```
```
model.add(Dense(char_to_num.vocabulary_size()+ 1 ,
kernel_initializer='he_normal', activation='softmax'))
model.summary()
```

Model: "sequential"
**_________________________________________________________________**
Layer (type) Output Shape Param #
=================================================================
conv3d (Conv3D) (None, 75, 46, 140, 128) 3584

```
activation (Activation) (None, 75, 46, 140, 128) 0
```
```
max_pooling3d (MaxPooling3D (None, 75, 23, 70, 128) 0
)
```
```
conv3d_1 (Conv3D) (None, 75, 23, 70, 256) 884992
```
```
activation_1 (Activation) (None, 75, 23, 70, 256) 0
```
```
max _pooling3d_ 1 (MaxPooling (None, 75, 11, 35, 256) 0
3D)
```
```
conv3d_2 (Conv3D) (None, 75, 11, 35, 75) 518475
```
```
activation_2 (Activation) (None, 75, 11, 35, 75) 0
```
```
max _pooling3d_ 2 (MaxPooling (None, 75, 5, 17, 75) 0
3D)
```
```
time_distributed (TimeDistr (None, 75, 6375) 0
ibuted)
```
```
bidirectional (Bidirectiona (None, 75, 256) 6660096
l)
```
```
dropout (Dropout) (None, 75, 256) 0
```
```
bidirectional_1 (Bidirectio (None, 75, 256) 394240
nal)
```
```
dropout_1 (Dropout) (None, 75, 256) 0
```
```
dense (Dense) (None, 75, 41) 10537
```
=================================================================
Total params: 8,471,924
Trainable params: 8,471,924
Non-trainable params: 0


**KeyComponents**

1. **ImportedLibraries** :
    ○ Sequential:Forstackinglayerssequentially.
    ○ Conv3D:3Dconvolutionallayer,idealforvideodata.
    ○ LSTM,Bidirectional:Forcapturingtemporalrelationships.
    ○ Dense:Fullyconnectedlayerforoutputprediction.
    ○ Dropout:Regularizationtopreventoverfitting.
    ○ MaxPool3D:Down-sampling3Dfeaturemaps.
    ○ Activation:Fornon-lineartransformations.
    ○ TimeDistributed:Applieslayerstoeachtimestepinasequence.
    ○ Flatten:Flattens3Dtensorsto2Dfordenselayers.
    ○ Adam:Anoptimizer.
    ○ ModelCheckpoint,LearningRateScheduler:Callbacksfortraining.
2. **InputDataShape** :
    ○ (75,46,140,1):
       ■ 75 framesinthevideo(timedimension).
       ■ 46x140resolutionforeachframe(heightxwidth).
       ■ 1 channel(grayscale).

**ModelArchitecture**

1. **InputandFirstConvolutionLayer** :
    ○ Conv3D(128,3,input_shape=(75,46,140,1),padding='same')
       ■ A3Dconvolutionwith 128 filtersandakernelsizeof3.
       ■ Inputshapeisspecifiedas(75,46,140,1).
    ○ Activation('relu'):AppliesReLUactivationfornon-linearity.
    ○ MaxPool3D((1,2,2)):Reducesspatialdimensions(heightand
       width)bypoolingwithastrideof 2 inspatialdimensions.
2. **SecondConvolutionLayer** :
    ○ Conv3D(256,3,padding='same'):Adeeperconvolutionwith 256
       filters.
    ○ MaxPool3D((1,2,2)):Furtherreducesspatialdimensions.
3. **ThirdConvolutionLayer** :
    ○ Conv3D(75,3,padding='same'):Asmallerconvolutionwith 75


```
filters,whichmayalignwiththetimedimension.
○ MaxPool3D((1,2,2)):Down-samplesfurther.
```
4. **TimeDistributedFlattening** :
    ○ TimeDistributed(Flatten()):Flattensspatialdimensionsateach
       timestep,resultingina1Dfeaturevectorforeachframe.
5. **BidirectionalLSTMLayers** :
    ○ **WhyLSTMs?** Theycapturetemporaldependenciesinsequential
       data.
    ○ Bidirectional(LSTM(128,kernel_initializer='Orthogonal',
       return_sequences=True)):
          ■ UsesbidirectionalLSTMstocapturetemporaldependencies
             inbothforwardandbackwarddirections.
          ■ return_sequences=True:OutputstheLSTMstatesforevery
             timestep.
          ■ Dropout(.5):Addsdropoutforregularization.
    ○ Repeatedforanadditionallayer.
6. **FinalDenseLayer** :
    ○ Dense(char_to_num.vocabulary_size()+1,activation='softmax'):
       ■ Fullyconnectedlayerwithoutputsizeequaltothe
          vocabularysize(char_to_num.vocabulary_size())plusone
          (likelyforablanktokenforCTCloss).
       ■ softmax:Outputsprobabilitiesforeachcharacterinthe
          vocabulary.

**ModelSummary**

```
● TotalParameters :8,471,924
● TrainableParameters :8,471,924(allaretrainable).
● Non-TrainableParameters : 0
```
**LayerOutputs**

1. InitialConv3D:(75,46,140,128)
2. After1stMaxPool3D:(75,23,70,128)
3. After2ndConv3D:(75,23,70,256)


4. After2ndMaxPool3D:(75,11,35,256)
5. After3rdConv3D:(75,11,35,75)
6. After3rdMaxPool3D:(75,5,17,75)
7. TimeDistributedFlatten:(75,6375)
8. BidirectionalLSTMLayers:(75,256)
9. FinalDenseLayer:(75,41)

```
yhat = model.predict(val[ 0 ])
```
```
tf.strings.reduce_join([num_to_char(x) for x in
tf.argmax(yhat[ 0 ],axis= 1 )])
```
```
tf.strings.reduce_join([num_to_char(tf.argmax(x)) for x
in yhat[ 0 ]])
```
```
model.input_shape
```
```
model.output_shape
```
Here’stheexplanationofthecodeyouprovided:

**MakingPredictions:**

```
yhat = model.predict(val[0])
```
Thislineusesthemodeltomakepredictionsontheinputdataval[0].Themodel
processesthisinputandgeneratesanoutput,whichisstoredinyhat.

**ProcessingPredictions:**

```
tf.strings.reduce_join([num_to_char(x) for x in
tf.argmax(yhat[ 0 ], axis= 1 )])
```
tf.argmax(yhat[0],axis=1)givestheindexofthemaximumvalueinyhat[0]
alongthespecifiedaxis(axis=1).Thisessentiallyprovidesthepredictedclass
foreachtimestepinthesequence.

num_to_char(x)convertsthepredictedclassindexintoacorresponding
character.


tf.strings.reduce_jointhenjoinsthesecharacterstogethertoformasinglestring.

**AlternativePredictionProcessing:**

```
tf.strings.reduce_join([num_to_char(tf.argmax(x)) for x
in yhat[ 0 ]])
```
Thislineprocesseseachtimestep’spredictioninyhat[0].Foreachtimestepx,
tf.argmax(x)findsthepredictedclassindex,whichisthenconvertedintoa
characterusingnum_to_char.

tf.strings.reduce_joinjoinsallthecharacterstoformastring.

**ModelInputandOutputShapes:**

```
model.input_shape
model.output_shape
```
model.input_shapereturnstheshapeoftheinputthatthemodelexpects.For
example,itcouldbe(None,75,46,140,1),indicatingabatchof3Dvideo
frameswithspecificdimensions.

model.output_shapereturnstheshapeoftheoutputthatthemodelproduces
afterprocessingtheinput.Forinstance,itcouldbe(None,75,41),whichmeans
themodelpredicts 41 classesforeachofthe 75 timestepsinthesequence.

Insummary,thiscodemakespredictions,processesthosepredictionsinto
readablestrings,andretrievesthemodel'sinputandoutputshapestounderstand
theexpecteddatadimensions.

#### SetupTrainingOptionsandTrain

```
def scheduler(epoch, lr):
if epoch< 30 :
return lr
else:
return lr* tf.math.exp(-0.1)
```
Thisfunctionisalearningratescheduler,whichadjuststhelearningrateduring
thetrainingprocessbasedonthecurrentepoch.Itistypicallyusedtoimprove
trainingefficiencyandhelpthemodelconvergemoreeffectively.


**FunctionDefinition:**

```
○ epoch:Thecurrenttrainingepoch(anintegervalue).
○ lr:Thecurrentlearningrate(afloatvalue).
```
**Condition(Epoch<30):**

```
if epoch < 30 :
return lr
```
Ifthecurrentepochislessthan30,thefunctionkeepsthelearningrate
unchanged.

Thisallowsthemodeltolearnsteadilywithaconstantlearningrateduringthe
initialtrainingphase.

**Condition(Epoch>=30):**

```
else:
return lr * tf.math.exp(-0.1)
```
Oncetheepochreachesorexceeds30,thelearningrateisreduced
exponentially.

tf.math.exp(-0.1)calculatestheexponentialdecayfactor(approximately
0.9048).

Thenewlearningrateissettothecurrentlearningrate(lr)multipliedbythis
decayfactor,causingthelearningratetodecreasegradually.

**WhyUseIt?**

```
● InitialStability: Aconstantlearningrateintheearlyepochsallowsthe
modeltofindagooddirectionforoptimizationwithoutdrasticchanges.
● Fine-Tuning: Asthelearningratedecreasesinlaterepochs,themodel
performssmallerupdatestoweights,whichhelpsrefinethelearned
featuresandavoidovershootingtheoptimalsolution.
```

**Example:**

Supposetheinitiallearningrateis0.001:

```
● Epoch10: Learningrateremains0.001.
● Epoch30: Learningratetransitionstoexponentialdecay.
○ Forthe30thepoch,thenewlearningratewouldbe:
```
```
0.001×e−0.1≈ 0.0009048
```
```
● Epoch50: Learningratecontinuestodecayfurther:
○ Forthe50thepoch:
```
```
0.001×(e−0.1)^20 ≈ 0.000163
```
Thisgradualreductioninlearningratehelpsimprovethemodel'sperformance
byfine-tuningweightsduringlaterepochs.

```
def CTCLoss(y_true, y_pred):
batch_len = tf.cast(tf.shape(y_true)[0],dtype="int64")
input_length = tf.cast(tf.shape(y_pred)[1],dtype="int64")
label_length = tf.cast(tf.shape(y_true)[1],dtype="int64")
input_length = input_length * tf.ones(shape=(batch_len, 1),
dtype="int64")
label_length = label_length * tf.ones(shape=(batch_len, 1),
dtype="int64")
```
```
loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length,
label_length)
return loss
```
TheCTCLoss(ConnectionistTemporalClassificationLoss)functioncalculates
thelossforsequencepredictiontaskswherethealignmentbetweeninputs
(predictions)andoutputs(groundtruths)isunknown.Itiscommonlyusedfor
taskslikespeech-to-text,OCR,orvideocaptioning.

**CodeBreakdown:**

```
def CTCLoss(y_true, y_pred):
```

```
● y_true :Thetruelabelsorsequences(groundtruth)representedas
numericalindices.
● y_pred :Thepredictedoutputs(logits)fromthemodel.Theseare
typicallyunalignedandrequiredecoding.
```
**1.BatchLengthCalculation:**

```
batch_len = tf.cast(tf.shape(y_true)[ 0 ], dtype="int64")
```
```
● tf.shape(y_true)[0]:Retrievesthenumberofsamples(batchsize)from
thegroundtruthtensor.
● tf.cast(...,dtype="int64"):Ensuresthevalueiscasttothecorrecttype
(int64)forcompatibilityinsubsequentoperations.
```
**2.InputLengthCalculation:**

```
input_length = tf.cast(tf.shape(y_pred)[ 1 ],
dtype="int64")
```
```
● tf.shape(y_pred)[1]:Retrievesthetimedimension(sequencelength)of
thepredictions.
● Caststheinputlengthtoint64.
```
**3.LabelLengthCalculation:**

```
label_length = tf.cast(tf.shape(y_true)[ 1 ],
dtype="int64")
```
```
● tf.shape(y_true)[1]:Retrievesthesequencelengthofthegroundtruth
labels.
● Caststhelabellengthtoint64.
```

**4.RepeatInputLengthsAcrossBatch:**

```
input_length = input_length * tf.ones(shape=(batch_len,
1 ), dtype="int64")
```
```
● tf.ones(shape=(batch_len,1),dtype="int64"):Createsatensorofshape
(batch_len,1)filledwithones,whereeachrowcorrespondstoasample
inthebatch.
● Multipliestheinputlengthscalarbythistensor,resultinginatensor
whereeachsampleinthebatchhasthesameinputlength.
```
**5.RepeatLabelLengthsAcrossBatch:**

```
label_length = label_length * tf.ones(shape=(batch_len,
1 ), dtype="int64")
```
```
● Similarly,createsatensorofrepeatedlabellengthsforeachsampleinthe
batch.
```
**6.ComputeCTCLoss:**

```
loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred,
input_length, label_length)
```
```
● ctc_batch_cost(...):ComputestheCTClossforeachsampleinthebatch.
○ Inputs :
■ y_true:Thetruelabels.
■ y_pred:Thepredictedoutputs.
■ input_length:Theactuallengthsofthepredictedsequences.
■ label_length:Theactuallengthsofthegroundtruth
sequences.
○ Thisfunctionalignsthepredictionstothelabelsandcalculatesthe
lossbasedontheprobabilityofthecorrectlabelsequence.
```

**7.ReturnLoss:**

```
return loss
```
```
● Thecomputedlosstensor(batch-wiselossvalues)isreturnedforusein
modeloptimization.
```
**WhyUseThisFunction?**

```
● TheCTClossisspecificallydesignedforsequencetaskswheretheinput
andoutputlengthsdifferandexplicitalignmentisnotprovided.
● Itenablesthemodeltopredictsequenceswithoutrequiring
frame-by-framelabeleddata.
```
```
class ProduceExample(tf.keras.callbacks.Callback):
def __init__(self,dataset) ->None:
self.dataset = dataset.as_numpy_iterator()
def on_epoch_end(self,epoch, logs=None)->None:
data = self.dataset.next()
yhat = self.model.predict(data[ 0 ])
decoded =tf.keras.backend.ctc_decode(yhat,[ 75 , 75 ],
greedy=False)[ 0 ][ 0 ].numpy()
for xinrange(len(yhat)):
print('Original:',
tf.strings.reduce_join(num_to_char(data[ 1 ][x])).numpy().decode('utf-8'))
print('Prediction:',
tf.strings.reduce_join(num_to_char(decoded[x])).numpy().decode('utf-8'))
print('~'* 100 )
```
HereisadetailedexplanationoftheProduceExampleclass:

**Purpose:**

TheProduceExampleclassisacustomcallbackinTensorFlow/Kerasthat:

1. Generatesandprintsexamplesofmodelpredictionsattheendofeach
    trainingepoch.


2. Allowsforreal-timemonitoringofthemodel'sperformanceby
    comparingpredictionswithgroundtruths.

Thisisparticularlyusefulintaskslikesequenceprediction,wheredecodingand
understandingpredictionsiscritical(e.g.,OCRorspeech-to-text).

**CodeBreakdown:**

**1.ClassDefinition:**

```
class ProduceExample(tf.keras.callbacks.Callback):
```
```
● Inheritsfromtf.keras.callbacks.Callback,abaseclassprovidedbyKeras
forcustomizingmodeltrainingbehavior.
● Youcanusethistodefinecustomactionslikelogging,savingmodels,or
evaluatingperformanceduringtraining.
```
**2.Initialization(__init__Method):**

```
def __init__(self, dataset) -> None:
self.dataset = dataset.as_numpy_iterator()
```
```
● dataset :ATensorFlowdataset(likelythetestorvalidationdataset)is
passedasinput.
● as_numpy_iterator() :ConvertsthedatasetintoaNumPyiterator,
allowingittobeiteratedusing.next().
```
Thisensureseasyaccesstoindividualbatchesofdataduringtraining.

**3.ActionattheEndofEachEpoch(on_epoch_endMethod):**

```
def on_epoch_end(self, epoch, logs=None) -> None:
```
```
● Thismethodiscalledautomaticallyattheendofeachepochduring
training.
● epoch :Thecurrentepochnumber.
```

```
● logs :Dictionaryoflogscontainingtraining/validationmetrics(e.g.,loss,
accuracy).
```
**4.FetchaBatchofData:**

```
data = self.dataset.next()
```
```
● Fetchesthenextbatchfromthedataset.
● data[0] :Inputfeatures(e.g.,imagesorsequences).
● data[1] :Groundtruthlabelsfortheinputs.
```
**5.ModelPredictions:**

```
yhat = self.model.predict(data[ 0 ])
```
```
● self.model :Referstothemodelbeingtrained.
● predict(data[0]) :Generatespredictionsfortheinputfeaturesinthe
batch.
```
**6.DecodePredictions:**

```
decoded = tf.keras.backend.ctc_decode(yhat, [75,75],
greedy=False)[ 0 ][ 0 ].numpy()
```
```
● ctc_decode :Decodesthepredictedsequencesintoreadabletextusingthe
CTCdecodingalgorithm.
○ yhat :Thepredictedlogits.
○ [75,75] :Specifiesthesequencelengthsforeachprediction(batch
size= 2 inthisexample).
○ greedy=False :Usesabeamsearchdecoder(betteraccuracythan
greedydecoding).
○ [0][0].numpy() :RetrievesthedecodedsequencesasaNumPy
array.
```

**7.IterateOverBatchandPrintResults:**

```
for x in range(len(yhat)):
print('Original:',
tf.strings.reduce_join(num_to_char(data[ 1 ][x])).numpy().d
ecode('utf-8'))
print('Prediction:',
tf.strings.reduce_join(num_to_char(decoded[x])).numpy().d
ecode('utf-8'))
print('~'* 100 )
```
```
● Loopthroughthebatch :
○ xiteratesoverallsamplesinthebatch.
● GroundTruth(Original) :
○ data[1][x]:Accessthetruesequence.
○ num_to_char(data[1][x]):Mapsthenumericallabelstocharacters.
○ tf.strings.reduce_join(...):Concatenatescharactersintoasingle
string.
○ .numpy().decode('utf-8'):Convertsthetensorintoa
Python-readableUTF-8string.
● Prediction(Prediction) :
○ Similardecodingprocessisappliedtothemodel'spredictions
(decoded[x]).
● PrintResults :
○ Printstheoriginalandpredictedsequences.
○ ~'*100':Printsaseparatorforbetterreadability.
```
**KeyFeatures:**

1. **Real-TimeFeedback** :Providesasnapshotofhowwellthemodelis
    learningateachepoch.
2. **Comparison** :Showsgroundtruthandpredictionssidebysideforeasy
    assessment.
3. **Debugging** :Helpsidentifycommonerrorsormismatchesinpredictions.
4. **CustomDecoding** :UsesCTCdecoding,makingitsuitablefortaskswith
    unalignedinput-outputsequence


```
model.compile(optimizer=Adam(learning_rate=0.0001),loss=CTCLoss)
```
configuresthemodelfortrainingbyspecifying:

1. **Theoptimizer** :Controlshowthemodelupdatesitsweightstominimize
    thelossfunction.
2. **Thelossfunction** :Measuresthedifferencebetweenthemodel’s
    predictionsandthegroundtruth.

**1.model.compile():**

```
● ThisisaKerasmethodtoconfigurethemodelfortraining.
● Itacceptsparameterslike:
○ Optimizer :Thealgorithmusedtoupdatemodelweights.
○ LossFunction :Afunctionusedtomeasuretheerrorduring
training.
○ Metrics :(Optional)Additionalmetricstoevaluatethemodel's
performance(notusedhere).
```
**2.optimizer=Adam(learning_rate=0.0001):**

```
● Adam :Standsfor AdaptiveMomentEstimation ,awidelyusedoptimizer
indeeplearning.
● LearningRate :Controlsthestepsizeforweightupdatesduringtraining.
Asmallvalue(0.0001)ensuresthatthemodeltrainsslowlyandavoids
overshootingtheminimumofthelossfunction.
● Adamcombinestheadvantagesoftwootheroptimizers:
```
1. **Momentum** :Usespastgradientstosmoothupdates.
2. **RMSProp** :Scalesupdatesbasedonrecentgradientmagnitudes.

**3.loss=CTCLoss:**

```
● The lossfunction usedhereisCTCLoss,whichstandsfor Connectionist
TemporalClassificationLoss.
● Purpose :
○ Designedforsequencepredictiontaskswheretheinputandoutput
```

```
sequencesarenotaligned.
○ Commonlyusedinspeech-to-text,OCR,orhandwriting
recognitiontasks.
● HowItWorks :
○ Penalizesthemodelforincorrectpredictionsbycomparingthe
predictedsequences(y_pred)withthegroundtruthsequences
(y_true).
○ Accountsforsequencealignmentissues(e.g.,missingcharactersor
extracharactersinpredictions).
```
**Summary:**

Thislineconfiguresthemodelfortrainingwith:

1. **Adamoptimizer** :Ensuresefficientandadaptiveweightupdates.
2. **CTCloss** :Handlessequencepredictiontaskswhereinput-output
    alignmentsarenotstrict.

Byspecifyingalearningrateof0.0001,themodelwilltraingradually,reducing
thelikelihoodofinstabilityorovershootingduringtraining.

```
checkpoint_callback =ModelCheckpoint(os.path.join('models','checkpoint'),
monitor='loss', save_weights_only=True)
```
Thiscodecreatesa **callback** thatsavesthemodel'sweightsduringtrainingat
specificintervals(usuallyattheendofeachepoch).Itisespeciallyusefulto:

1. Saveintermediatetrainingprogress.
2. Resumetrainingincaseofinterruptions.
3. Preventlossofmodelweightsaftertraining.

**1.ModelCheckpoint:**

```
● Abuilt-inKerascallbackusedtosavethemodelduringtraining.
● Parameters:
○ filepath :Specifieswhereandhowtosavethemodelweights.
○ monitor :Metrictomonitorforsavingthecheckpoint(e.g.,loss,
```

```
accuracy).
○ save_weights_only :Whethertosavejusttheweightsortheentire
model(weights+architecture).
○ Otheroptions(notusedhere):
■ save_best_only:Saveonlythebestweightsbasedonthe
monitoredmetric.
■ mode:Specifieswhethertolookfortheminimumor
maximumvalueofthemonitoredmetric.
```
**2.os.path.join('models','checkpoint'):**

```
● Constructsafilepathtosavethecheckpoints.
● os.path.join :
○ Combinesdirectorypaths('models')andfilenames('checkpoint').
○ Ensurescompatibilityacrossdifferentoperatingsystems.
● Here,thecheckpointswillbesavedinthemodelsdirectorywiththe
filenamecheckpoint.
```
**3.monitor='loss':**

```
● Specifiesthatthe loss metricismonitoredduringtraining.
● Thecallbackwillsavethemodel'sweightsaftereachepoch,regardlessof
whetherthelossimproves(unlesssave_best_only=Trueisset).
```
**4.save_weights_only=True:**

```
● Indicatesthat onlythemodelweights shouldbesaved,notthefullmodel
structure.
● Advantage:
○ Savesstoragespace.
○ Youcanloadtheweightslaterintoamodelwiththesame
architecture.
```
**ExampleScenario:**

```
● DuringTraining :Aftereachepoch,thiscallbacksavesthemodel's
weightsintothefilemodels/checkpoint.
● UseCase :Iftrainingisinterrupted,youcanreloadthesavedweightsinto
```

```
thesamemodelandcontinuetrainingwithoutstartingover.
```
**Summary:**

ThislinedefinesaModelCheckpointcallbackthatsavesthemodel'sweights
aftereveryepoch.Itmonitorsthe **loss** metricandstorestheweightsinthe
modelsdirectoryunderthefilenamedcheckpoint.Bysavingonlytheweights,
itensuresefficientstorageandquickrecoveryoftrainingprogress.

```
schedule_callback = LearningRateScheduler(scheduler)
```
Thiscodecreatesa **callback** thatadjuststhelearningratedynamicallyduring
trainingusingthe **scheduler** function.Thishelpsoptimizethemodel'straining
byvaryingthelearningratebasedontheepoch.

**1.LearningRateScheduler:**

```
● AKerascallbackthatupdatesthelearningrateoftheoptimizerduring
training.
● Parameters:
○ schedule :Afunctionthatdefineshowthelearningrateshould
changeoverepochs.Thisfunctionisexecutedatthestartofevery
epoch.
○ verbose :(optional)Ifsetto1,itprintstheupdatedlearningrateat
eachepoch.
```
**2.scheduler:**

```
● Thefunctionprovidedtothescheduleparameter.
```
Theschedulerfunctioninyourcodeisdefinedas:

```
def scheduler(epoch, lr):
if epoch < 30 :
return lr
else:
return lr * tf.math.exp(-0.1)
```

```
● Purpose :
○ Forthefirst 30 epochs:Keepsthelearningrateconstant.
○ After 30 epochs:Graduallydecreasesthelearningrate
exponentiallybymultiplyingthecurrentlearningratewithe−0.1.
```
**3.schedule_callback:**

```
● AvariablethatholdstheLearningRateSchedulerinstance.
● Thiscallbackislaterpassedtothemodel.fit()methodtoapplythe
learningratescheduleduringtraining.
```
**WhyUseaLearningRateScheduler?**

```
● FasterConvergence :Ahighlearningrateinearlyepochsaccelerates
progress.
● Stability :Alowerlearningrateinlaterepochspreventsovershootingthe
optimalpoint.
● AvoidsOverfitting :Gradualreductioninlearningratehelpsrefinethe
model'sparametersandachievebettergeneralization.
```
**ExampleScenario:**

```
● DuringTraining :
○ Inepochs0–29:Thelearningrateremainsconstant(e.g.,0.001if
theinitiallearningrateissettothisvalue).
○ Fromepoch 30 onwards:Thelearningratedecaysexponentially
(e.g.,lr=lr×e−0.1).
```
**Summary:**

Theschedule_callbackadjuststheoptimizer'slearningratedynamicallybased
ontheschedulerfunctionduringtraining.Thisensuresaneffectivetraining
processbymaintainingahighlearningrateinitiallyandreducingitinlater
epochsforbetterfine-tuning.


```
example_callback = ProduceExample(test)
```
Thiscodecreatesaninstanceofthe **ProduceExample** class,whichisacustom
callbackdefinedearlier.Thiscallbackgeneratespredictionsandcomparesthem
totheoriginallabelsaftereachepochduringmodeltraining.It'sprimarilyused
formonitoringthemodel'sperformancevisuallyontheprovidedtestdataset.

**1.ProduceExample:**

```
● Thisisacustomcallbackclassinheritingfrom
tf.keras.callbacks.Callback.
```
Definedas:

```
class ProduceExample(tf.keras.callbacks.Callback):
def __init__(self, dataset) -> None:
self.dataset= dataset.as_numpy_iterator()
```
```
def on_epoch_end(self, epoch, logs=None) -> None:
data = self.dataset.next()
yhat = self.model.predict(data[ 0 ])
decoded = tf.keras.backend.ctc_decode(yhat, [ 75 , 75 ],
greedy=False)[ 0 ][ 0 ].numpy()
for x in range(len(yhat)):
print('Original:',
tf.strings.reduce_join(num_to_char(data[ 1 ][x])).numpy().decode('utf-8'))
print('Prediction:',
tf.strings.reduce_join(num_to_char(decoded[x])).numpy().decode('utf-8'))
print('~'* 100 )
```
```
● Functionality :
```
```
○ Attheendofeachtrainingepoch(on_epoch_endmethod):
■ Fetchesabatchofdatafromtheprovideddataset.
■ Runsthemodel'spredictionontheinputdata.
■ Decodesthepredictions(likelyusingConnectionist
TemporalClassification,orCTC).
■ Printstheoriginallabelsandpredictedoutputsforvisual
comparison.
```

**2.test:**

```
● Thisisthedatasetpassedtothecallback.Itcontainsdatathemodelhas
notseenduringtraining.
● LikelypreparedusingTensorFlow'stf.dataAPIorasimilarmethod.
● Examplecontentoftest:
○ Inputdata :Abatchoffeaturearrays(e.g.,images,videoframes).
○ Labels :Thetruesequences/outputscorrespondingtotheinputs.
```
**3.example_callback:**

```
● ThevariableholdinganinstanceofProduceExample,initializedwiththe
testdataset.
● Thisinstancewilllaterbepassedtothemodel.fit()methodtoexecuteits
functionalityduringtraining.
```
**WhyUseProduceExample?**

```
● MonitorTrainingProgress :
○ Outputspredictionsandtheircorrespondingtruelabelsforthetest
dataset.
○ Allowsreal-timeevaluationofhowwellthemodelislearning.
● Debugging :
○ Ifpredictionsdon'timproveoralignwithlabels,adjustmentscan
bemadetothemodelortrainingprocess.
```
**Summary:**

Theexample_callbackisacustomcallbackinstancethatfetchesdatafromthe
testdataset,performspredictionsusingthemodel,decodesthepredictions,and
printsthemalongsidethetruelabelsattheendofeachepoch.Itprovidesan
intuitivewaytomonitormodelprogressandensureitlearnsasexpectedduring
training.


```
model.fit(train, validation_data=test,epochs=100,
callbacks=[checkpoint_callback, schedule_callback, example_callback])
```
Thecodetrainsthe **model** usingthe **train** dataset,validatesitagainstthe **test**
dataset,andappliesspecified **callbacks** tomonitorandmodifythetraining
process.Itrunsfor 100 epochs.

**1.model.fit():**

```
● ThismethodtrainstheKerasmodel.
● Keyarguments:
○ train :Thetrainingdataset,typicallypreparedusingTensorFlow's
tf.dataAPIoranotherdatapreprocessingpipeline.
○ validation_data=test :Aseparatedatasetusedtoevaluatethe
model'sperformanceattheendofeachepoch.Helpsdetect
overfittingorunderfitting.
○ epochs=100 :Thenumberofcompletepassesthroughthetraining
datasetduringtraining.
○ callbacks :Alistofcallbackobjectsformonitoringandmanaging
training.
```
**2.train:**

```
● Theinputdatasetcontainsfeatures(e.g.,images,sequences)andtheir
correspondinglabels.
● Typicallypreprocessed(e.g.,scaled,augmented)andbatched.
```
**3.validation_data=test:**

```
● The test datasetservesasavalidationset.
● Helpsevaluatethemodel'sperformanceonunseendataaftereachepoch.
```
**4.epochs=100:**

```
● Thetrainingprocesswilliteratethroughtheentire train dataset 100
times.
```
**5.callbacks:**


```
● Alistofcallbackobjectsthatareexecutedduringtrainingatspecific
points.Thesecallbackscansavemodels,adjustlearningrates,ormonitor
performance.
```
```
○ Checkpoint_callback :
■ Savesthemodel'sweightsattheendofeachepochifthe
monitoredmetric(e.g.,loss)improves.
```
Codeusedtodefineit:

```
checkpoint_callback = ModelCheckpoint( os .path.join('models',
'checkpoint'), monitor='loss', save_weights_only=True)
```
```
○ Schedule_callback :
■ Adjuststhelearningratedynamicallybasedontheepoch
number.
■ Usesthecustomschedulerfunctiontoreturnamodified
learningrateaftereachepoch.
```
Codeusedtodefineit:
schedule_callback = LearningRateScheduler(scheduler)

```
○ example_callback :
```
```
■ Monitorspredictionsvisuallybycomparingthemodel's
predictionswiththetruelabelsattheendofeachepoch.
■ Customcallbackthatprovidesdebuggingandmonitoringfor
themodel'slearningprocess.
```
Codeusedtodefineit:

```
example_callback = ProduceExample( test )
```
**KeyPoints:**

```
● TrainingProcess :
```
```
○ Themodeliteratesoverthe train datasetandcomputesgradientsto
```

```
optimizeitsparameters.
○ Aftereachepoch,themodelevaluatesitsperformanceonthe test
dataset.
● CallbacksinAction :
```
```
○ Thecheckpoint_callbacksavesthemodel’sprogressbasedonits
performance.
○ Theschedule_callbackadjuststhelearningratedynamicallyfor
improvedconvergence.
○ Theexample_callbackoutputssamplepredictionsformanual
inspectionoftrainingprogress.
```
**Summary:**

Thislinetrainsthemodelfor 100 epochsusingthetraindatasetwhilevalidating
onthetestdataset.Thetrainingprocessisaugmentedwithcallbacksto:

1. Savethebestmodelweights(checkpoint_callback),
2. Dynamicallyadjustthelearningrate(schedule_callback),and
3. Printpredictionsfordebuggingandmonitoring(example_callback).

### MakeaPrediction

```
url = 'https://drive.google.com/uc?id=1vWscXs4Vt0a_1IH1-ct2TCgXAZT-N3_Y'
output ='checkpoints.zip'
gdown.download(url, output, quiet=False)
gdown.extractall('checkpoints.zip','models')
```
**1.url = 'https://drive.google.com/uc?id=1vWscXs4Vt0a_1IH1-ct2TCgXAZT-N3_Y':**

```
● ThisdefinestheURLofthefiletobedownloaded.
● TheURLpointstoafilehostedonGoogleDrive.
○ uc?id=1vWscXs4Vt0a_1IH1-ct2TCgXAZT-N3_Y :Theunique
identifierforthefileonGoogleDrive.
```
**2.output='checkpoints.zip':**

```
● Thisspecifiesthenameofthefilewherethedownloadeddatawillbe
```

```
savedlocally.
● Thefilewillbesavedas checkpoints.zip.
```
**3.gdown.download(url,output,quiet=False):**

```
● gdown :APythonlibrarythatsimplifiesdownloadingfilesfromGoogle
Drive.
● gdown.download() :
○ url :TheURLofthefiletodownload(providedabove).
○ output :Thenameoftheoutputfilewherethedownloadedcontent
willbesaved.
○ quiet=False :Ensuresprogressisdisplayedintheconsoleduring
thedownload.
```
**4.gdown.extractall('checkpoints.zip','models'):**

```
● gdown.extractall() :
○ Extractsthecontentsofthezipfile checkpoints.zip.
○ Firstargument('checkpoints.zip') :
■ Pathtothezipfiletobeextracted.
○ Secondargument('models') :
■ Thedestinationfolderwheretheextractedfileswillbe
placed.
■ Inthiscase,thecontentsofthezipfilewillbeextractedinto
afoldernamed models.
```
**Workflow:**

1. **Download** :

```
○ ThefileidentifiedbytheURLisdownloadedfromGoogleDrive.
○ Itissavedlocallyas checkpoints.zip.
```
2. **Extract** :

```
○ Thedownloadedzipfileisextractedintothe models directory.
○ Thisunzippingstepiscrucialforaccessingthecontentsofthefile.
```

```
model.load_weights('models/checkpoint')
```
Thelinemodel.load_weights('models/checkpoint')loadsthesavedweightsofa
pre-trainedmodel.Theweightsareloadedfromthespecifiedpath
'models/checkpoint'.Thisallowsthemodeltoresumetrainingorperform
inferenceusingthesavedparameters.Theweightsarestoredinthecheckpoint
file,whichcontainsthelearnedparametersfromaprevioustrainingsession.

```
test_data = test.as_numpy_iterator()
```
Thelinetest_data=test.as_numpy_iterator()convertsthetestdatasetintoa
NumPyiterator.Thisallowsforiteratingthroughthetestdatasetinaformatthat
returnsNumPyarrays,makingiteasiertohandleduringmodelevaluationor
inference.Byusinganiterator,thedatasetcanbeprocessedinbatchesoneata
time,whichismemoryefficient.Itfacilitatesaccessingandusingthedataina
sequentialmannerforoperationslikepredictionorevaluation.

```
sample =test_data.next()
```
Thelinesample=test_data.next()retrievesthenextbatchofdatafromthe
test_dataiterator.Thismeansitfetchesthenextelementfromthetestdatasetas
aNumPyarray.Thenext()functionadvancestheiteratorandreturnsthenext
availablesample,whichistypicallyatuplecontainingtheinputfeaturesandthe
correspondinglabels.Thisisusefulforprocessingandmakingpredictionson
thetestdatasetduringmodelevaluationorinference.

```
yhat = model.predict(sample[ 0 ])
```
Thelineyhat=model.predict(sample[0])usesthetrainedmodeltomake
predictionsontheinputdatafromthesample.Here'sabreakdown:

1. sample[0]:Thisreferstotheinputfeatures(data)ofthecurrentsample
    fromthetest_data.Typically,sample[0]containsthefeatures(likeimages
    orsequences)thatthemodelwillpredict.
2. model.predict():Thismethodisusedtogeneratepredictionsbasedonthe
    inputdata.Itpassessample[0]throughthemodelandoutputspredicted


```
values.
```
3. yhat:Thisstoresthemodel'spredictionsfortheinputsample.These
    predictionscanthenbeprocessedfurther,suchasdecodingorevaluating
    themodel'sperformance.

```
print('~'* 100 , 'REALTEXT')
[tf.strings.reduce_join([num_to_char(word)for wordin sentence])for
sentence in sample[ 1 ]]
```
Thecodeyouprovidedprintsalineof 100 tildecharactersfollowedbythelabel
"REALTEXT"andthenattemptstotransformthelabelsfromthesample[1]
(presumablythetruelabelsofthedataset)intohuman-readabletext.Here'sa
detailedbreakdown:

1. print('~'*100,'REALTEXT'):
    ○ Thisprintsaseparatorline(~repeated 100 times)followedbythe
       text"REALTEXT".It'susedasavisualmarkertoseparateoutputs
       forclarity.
2. [tf.strings.reduce_join([num_to_char(word)forwordinsentence])for
    sentenceinsample[1]]:
       ○ Thislineisalistcomprehensionthatoperatesoneachsentencein
          sample[1](thetruelabelsorgroundtruth).Eachsentenceis
          transformedby:
             ■ [num_to_char(word)forwordinsentence]:Thisconverts
                eachword(whichisassumedtoberepresentedasnumeric
                valuesinthedataset)intocharactersusingthenum_to_char
                function.
             ■ tf.strings.reduce_join(...):Joinsthecharactersintoasingle
                string(sentence)foreachsentenceinsample[1].
       ○ Theresultisalistofdecodedsentences.
3. Thelistcomprehensionreturnsalistofhuman-readabletextsentences
    fromthegroundtruthlabels,whichisprintedfollowingtheseparator.

```
decoded = tf.keras.backend.ctc _decode(yhat, input_ length=[75,75],
greedy=True)[ 0 ][ 0 ].numpy()
```

Thecodedecoded=tf.keras.backend.ctc_decode(yhat,input_length=[75,75],
greedy=True)[0][0].numpy()performsthefollowingoperations:

1. tf.keras.backend.ctc_decode(yhat,input_length=[ 75 , 75 ], greedy=True):
    ○ ThisfunctiondecodestheoutputofaCTC(Connectionist
       TemporalClassification)model.
    ○ yhat:Thisisthemodel'spredictedoutput,typicallyaprobability
       distributionovercharactersforeachtimestepinthesequence.
    ○ input_length=[75,75]:Thelengthoftheinputsequencesinthe
       batch.It'sspecifiedas[75,75],meaningtheinputsequencesfor
       bothitemsinthebatchareoflength75.
    ○ greedy=True:Thisargumentspecifiesthatthegreedydecoding
       algorithmisused,meaningthemostprobablecharacterateach
       timestepischosen(asopposedtobeamsearchorotherdecoding
       methods).
    ○ Thectc_decodefunctionreturnsalistofdecodedsequences.Each
       sequencerepresentsthedecodedoutputofthemodelforeachinput
       sequence.
2. **[0][0]** :

```
○ Sincectc_decodereturnsalistoflists,[0][0]accessesthefirst
sequenceinthebatchandthefirstdecodedsentencewithinthat
sequence.
```
3. **.numpy()** :

```
○ ConvertstheTensorFlowtensortoaNumPyarray,allowingfurther
operationsormanipulationinNumPyformat(orforeasier
inspection).
```
TheresultstoredindecodedisthedecodedoutputoftheCTCmodelforthe
batch'sfirstsequence,whichrepresentsthemostlikelycharacterspredictedby
themodelforthatsequence.

```
print('~'* 100 , 'PREDICTIONS')
[tf.strings.reduce_join([num_to_char(word)for wordin sentence])for
```

```
sentence in decoded]
```
Thelineofcodeprint('~'*100,'PREDICTIONS')followedby
[tf.strings.reduce_join([num_to_char(word)forwordinsentence])forsentence
indecoded]performsthefollowingsteps:

1. print('~'* 100 , 'PREDICTIONS'):
    ○ Thisprintsaseparatorlineof 100 ~charactersfollowedbythe
       label'PREDICTIONS'totheconsole.Itservesasavisualmarker
       toindicatethestartofthepredictedoutputs.
2. [ **tf.strings.reduce_join** ([ **num_to_char** ( **word** ) for word in sentence])
    for sentence in decoded]:
       ○ Thisisalistcomprehensionthatprocesseseachsentencein
          decoded.
       ○ decoded:Itisassumedtobealistofdecodedsequences,where
          eachsequenceisalistofintegerindicesrepresentingthepredicted
          characters(asnumbers).
3. **[num_to_char(word)forwordinsentence]** :

```
○ Foreachsentence(whichisalistofpredictedcharacterindices),it
convertseachindex(word)toitscorrespondingcharacterusingthe
num_to_charfunction.Thistransformsthesequenceofnumbers
intoasequenceofcharacters.
```
4. **tf.strings.reduce_join(...)** :
    ○ tf.strings.reduce_jointakesthelistofcharactersforasentenceand
       joinsthemintoasinglestring(removingspacesbetweenthem).
    ○ Iteffectivelyconvertsalistofcharactersintoacontinuousstring
       representingthedecodedsentence.
5. **Theentirelistcomprehension** :
    ○ Itappliesthereduce_joinoperationtoeachsentencedecodedto
       createthecompletepredictedtextforeachinputsequence.

Theresultisalistofdecodedtextpredictions,whereeachpredictionisastring
ofcharactersrepresentingthemodel'soutputforeachinputsequence.Thisis
printedaftertheseparatorline.


##### TestonaVideo

```
sample =load_data(tf.convert_to_tensor('./data/s1/bras9a.mpg'))
print('~'* 100 , 'REALTEXT')
[tf.strings.reduce_join([num_to_char(word)for wordin sentence])for
sentence in [sample[ 1 ]]]
yhat = model.predict(tf.expand_dims(sample[ 0 ], axis= 0 ))
decoded = tf.keras.backend.ctc_decode(yhat,input_length=[ 75 ],
greedy=True)[ 0 ][ 0 ].numpy()
print('~'* 100 , 'PREDICTIONS')
[tf.strings.reduce_join([num_to_char(word)for wordin sentence])for
sentence in decoded]
```
Here’stheexplanationforthecodesnippetyouprovided:

**1.Loadingthevideosample:**

```
sample = load_data(tf.convert_to_tensor('./data/s1/bras9a.mpg'))
```
```
● Theload_datafunctionloadsthevideofilebras9a.mpglocatedat
'./data/s1/'andconvertsthepathtoatensorusingtf.convert_to_tensor().
Itassumesthefunctionprocessesthevideoandreturnsasample,
typicallycontainingboththevideoframesandthecorrespondingground
truthtext(labels).
```
**2.Printingtherealtext:**

```
print('~'* 100 , 'REAL TEXT')
[tf.strings.reduce_join([num_to_char(word) for word in
sentence]) for sentence in [sample[ 1 ]]]
```
```
● Thisprintsthelabel(therealtranscriptionorgroundtruth)forthevideo
sequence.
● sample[1]referstotherealtextinthesample,whichisconvertedto
charactersusingnum_to_char.Thereduce_joinfunctionthenjoinsthelist
ofcharactersintoastring,representingtherealtext.
● Theresultistheactualtextcorrespondingtothevideo,displayedaftera
separatorlineof 100 ~characters.
```

**3.Makingpredictions:**

```
yhat = model.predict(tf.expand_dims(sample[ 0 ], axis= 0 ))
```
```
● sample[0]containsthevideoframes(inputfeatures),andtf.expand_dims
addsanextradimensiontomatchthemodel'sexpectedinputshape.
● model.predict()runsinferenceontheinputsample,generatingthe
predictedoutputforthevideo,whichisstoredinyhat.
```
**4.DecodingthepredictionsusingCTC(ConnectionistTemporal
Classification):**

```
decoded = tf.keras.backend.ctc _decode(yhat,
input_ length=[ 75 ], greedy=True)[ 0 ][ 0 ].numpy()
```
```
● ctc_decodeisusedtodecodethemodel'soutput(yhat)intoactualtext
predictionsusingtheCTClossfunction.
● input_length=[75]indicatesthesequencelengthfortheinput(likely
basedonthelengthofthevideoframes).
● greedy=Trueensuresthatthedecodingisdoneusingthegreedyapproach,
wherethemostprobablecharacterischosenateachtimestep.
● The[0][0]accessesthefirstdecodingresult(asctc_decodereturnsalist
ofdecodedsequences).
● The.numpy()convertsthetensortoaNumPyarray.
```
**5.Printingthepredictions:**

```
print('~'* 100 , 'PREDICTIONS')
[tf.strings.reduce_join([num_to_char(word) for word in
sentence]) for sentence in decoded]
```
```
● Printsaseparatorlinefollowedbythepredictedtext.
● Eachdecodedsequence(sentenceindecoded)isconvertedfromindices
tocharactersusingnum_to_char,andreduce_joinjoinsthemintoasingle
string,representingthepredictedtranscription.
● Theresultisthepredictedtranscriptionofthevideo,shownasatext
string.
```

**Output:**

```
PREDICTIONS
[<tf.Tensor: shape=(), dtype=string, numpy=b'bin red at s nine
again'>]
```
```
● Themodelpredictsthetranscriptionas"binredatsnineagain",whichis
theoutputofthevideosamplebras9a.mpg.
```
Thisdemonstratesafullcycleofloadingavideo,makingpredictions,and
decodingtheresultsusingaCTC-basedmodelforvideocaptioningorspeech
recognition.

Thankyouforreading!


