import EmoMap.coling18.framework.prepare_data as data
import EmoMap.coling18.framework.models
from EmoMap.coling18.framework.reference_methods.aicyber import MLP_Ensemble
from keras.optimizers import Adam


VLIMIT = None

VAD = ['Valence', 'Arousal', 'Dominance']
VA = ['Valence', 'Arousal']
BE5 = ['Joy', 'Anger', 'Sadness', 'Fear', 'Disgust']


DIRECTIONS = {'vad2be': {'source': VAD, 'target': BE5},
			  'be2vad': {'source': BE5, 'target': VAD}}

MY_MODEL = {	'activation': 'relu',
			 'dropout_hidden': .2,
			 'train_steps': 10000,
			 'batch_size': 128,
			 'optimizer': Adam(lr=1e-3),
			 'hidden_layers': [128, 128]}

class Setting():
	def __init__(self, name, language, data, left, right, emo_type):
		'''
		Args:
				name 			String (describing results_file)
				language 		String (will be used to load embeddings)
				data 			function
		'''
		self.name = name
		self.language = language.lower()
		self.load_data = data
		self.left = left
		self.right = right
		self.emo_type = emo_type

	def get_left(self):
		inner = self.load_data()
		left = self.left().join(self.right(), how='left')
		vad_only = left[~left.index.isin(list(inner.index))]
		return vad_only[self.emo_type]

	def get_right(self):
		inner = self.load_data()
		right = self.left().join(self.right(), how='right')
		be_only = right[~right.index.isin(list(inner.index))]
		return be_only[BE5]

SETTINGS = [
	Setting('English_ANEW_Stevenson',
			'english',
			data.get_english_anew,
			data.load_anew10,
			data.load_stevenson07,
			VAD),
	Setting('English_Warriner_Stevenson',
			'english',
			data.get_english_warriner,
			data.load_warriner13,
			data.load_stevenson07,
			VAD),
	Setting('Spanish_Redondo',
			'spanish',
			data.get_spanish_redondo,
			data.load_redondo07,
			data.load_ferre16,
			VAD),
	Setting('Spanish_Hinojosa',
			'spanish',
			data.get_spanish_hinojosa,
			data.load_hinojosa16,
			data.load_hinojosa16,
			VAD),
	Setting('Spanish_Stadthagen',
			'spanish',
			data.get_spanish_stadthagen,
			data.load_stadthagen17,
			data.load_stadthagen17,
			VA),
	Setting('German_BAWL',
			'german',
			data.get_german_bawl,
			data.load_vo09,
			data.load_briesemeister11,
			VA),
	Setting('Polish_NAWL',
			'polish',
			data.get_polish_nawl,
			data.load_riegel15,
			data.load_wierzba15,
			VA),
	Setting('Polish_Imbir',
			'polish',
			data.get_polish_imbir,
			data.load_imbir16,
			data.load_wierzba15,
			VAD)
]

IN_PAPER_NAMES = {'English_ANEW_Stevenson': 'en_1',
				  'English_Warriner_Stevenson': 'en_2',
				  'Spanish_Redondo': 'es_1',
				  'Spanish_Hinojosa': 'es_2',
				  'Spanish_Stadthagen': 'es_3',
				  'German_BAWL': 'de_1',
				  'Polish_NAWL': 'pl_1',
				  'Polish_Imbir': 'pl_2'}

SHORT_COLUMNS = {'Valence': 'Val',
				 'Arousal': 'Aro',
				 'Dominance': 'Dom',
				 'Joy': 'Joy',
				 'Anger': 'Ang',
				 'Sadness': 'Sad',
				 'Fear': 'Fea',
				 'Disgust': 'Dsg'}

LANGUAGES = {setting.language for setting in SETTINGS}


def GET_EMBEDDINGS(language, vocab_limit=VLIMIT):
	return data.get_facebook_fasttext_wikipedia(language, vocab_limit)



KFOLD = 10
