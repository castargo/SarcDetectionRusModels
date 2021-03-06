import os


if 'SARCDET_INTERJECTIONS' in os.environ:
    interjections = set(os.environ.get('SARCDET_INTERJECTIONS', '').split(','))
else:
    interjections = {
        'а', 'ае', 'ай', 'але', 'алло', 'аминь', 'ась', 'атас', 'ау', 'ах', 'ахти', 'ахтунг',
        'ба', 'баста', 'батюшки', 'бац', 'бе', 'бинго', 'бис', 'бла', 'блин', 'бля', 'боже',
        'бр', 'браво', 'брейк', 'брысь', 'буэ', 'вау', 'вах', 'вон', 'вуаля', 'гип', 'гля', 
        'гм', 'гмм', 'го', 'гой', 'гоп', 'господи', 'дык', 'ё', 'ёб', 'еб', 'ей', 'еклмн', 
        'ёклмн', 'епрст', 'ёпрст', 'ёлы', 'ёпт', 'епт', 'жесть', 'жиза', 'йоу', 'ко', 'кусь', 
        'кхм', 'кыш', 'ля', 'м', 'ммм', 'мда', 'мля', 'мяу', 'нда', 'нате', 'ни', 'ну', 'нука', 
        'ня', 'о', 'ого', 'огого', 'ооо', 'окей', 'оба', 'обана', 'ого', 'ой', 'ойой', 'ок', 
        'окей', 'оп', 'опа', 'опаньки', 'оу', 'ох', 'охуенно', 'пиздец', 'поди', 'покеда', 
        'превед', 'привет', 'прочь', 'псс', 'пшш', 'салам', 'салют', 'слышь', 'сори', 'сорри', 
        'сорян', 'стопэ', 'сцуко', 'тадам', 'тсс', 'тьфу', 'тю', 'увы', 'угу', 'упс', 'ура', 
        'уря', 'уси', 'уф', 'ух', 'фак', 'фас', 'фи', 'фу', 'фью', 'хайль', 'хана', 'хаюшки', 
        'хвать', 'хм', 'хой', 'хорэ', 'хр', 'хрр', 'хрясь', 'хуяк', 'цап', 'цок', 'цыц', 'чао', 
        'шшш', 'ша', 'э', 'эээ', 'эй', 'эге', 'эгеге', 'эва', 'эврика', 'эге', 'эгей', 'эй', 'эх'
        # 'aah', 'ah', 'aha', 'ahem', 'alas', 'argh', 'aw', 'aww', 'bah', 'behold', 'bingo', 'boo', 
        # 'bravo', 'brr', 'dear', 'dah', 'duh', 'eek', 'eh', 'er', 'eww', 'gah', 'gee', 'grr', 'hah', 
        # 'hm', 'hmm', 'hey', 'huh', 'humph', 'hurrah', 'meh', 'mhm', 'muahaha', 'nuh', 'oh', 'ooh', 
        # 'la', 'oomph', 'oops', 'oww', 'ouch', 'oy', 'pew', 'pff', 'phew', 'psst', 'sheesh', 'shh', 
        # 'shoo', 'tsk', 'uh', 'uhh', 'ugh', 'um', 'umm', 'wee', 'well', 'whoa', 'wow', 'yahoo', 
        # 'yay', 'ye', 'yea', 'yeah', 'yikes', 'yippee', 'yoo', 'yuh', 'yuck', 'zing'
    }

if 'SARCDET_FUNNY_MARKS' in os.environ:
    funny_marks = set(os.environ.get('SARCDET_FUNNY_MARKS', '').split(','))
else:
    funny_marks = {
        'ха', 'хаха', 'хахаха', 'хах', 'хи', 'хихи', 'хе', 'хехе', 'хех', 'хохо', 'лол', 
        'ржу', 'ржунимагу', 'кек', 'бугага', 'охо', 'охохо', 'хо', 'баян', 'лойс', 'ору', 
        'рофл', 'омг'
        # 'ha', 'haha', 'lol', 'lolz', 'lmao', 'lqtm', 'lsmh', 'lmho', 'omg', 'rofl', 'lmfao'
    }
