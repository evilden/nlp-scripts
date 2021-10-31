import sys
import fasttext
import warnings
import pycountry

fasttext.FastText.eprint = lambda x: None
warnings.filterwarnings("ignore")
USAGE = f"Usage: python {sys.argv[0]} (--help | <path-to-test-csv-file>)\n" \
        f"Example: python detect_lang.py \"Умом Россиию не понять\""


class LanguageDetector:

    def __init__(self):
        pretrained_lang_model = "lid.176.bin"
        self.model = fasttext.load_model(pretrained_lang_model)

    def predict_lang(self, text):
        predictions = self.model.predict(text, k=1)  # returns top 2 matching languages
        lang_code = predictions[0][0][9:]
        if len(lang_code) == 2:
            lang_name = pycountry.languages.get(alpha_2=lang_code).name
        elif len(lang_code) == 3:
            lang_name = pycountry.languages.get(alpha_3=lang_code).name
        else:
            raise ValueError('Incorrect code length')
        return lang_name


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit(USAGE)

    if args[0] == "--help":
        print(USAGE)
        return
    else:
        data_to_predict_on = args[0]

    detector = LanguageDetector()
    lang_name = detector.predict_lang(data_to_predict_on)
    print(lang_name)
    return 0


if __name__ == '__main__':
    sys.exit(main())
