"""
Khoja Dictionary-Based Root Stemmer (System S4).

Extracts Arabic morphological roots and validates candidate roots
against a canonical 4,748 Arabic root lexicon (3,822 triliteral + 926 quadriliteral).

Reference:
    Khoja, S., & Garside, R. (1999).
    Stemming Arabic text.
    Computing Department, Lancaster University, Lancaster, UK.
"""

import os
import gzip
import base64
from typing import Optional, Set, Tuple, List

# Embedded 15.2 KB Base64-Gzipped canonical 4,748 Arabic root lexicon
# Decompresses in < 2ms to provide 100% offline self-contained execution.
EMBEDDED_KHOJA_ROOTS_B64 = "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"

# Static morphological tables derived from Khoja (1999)
DEFINITE_ARTICLES: Tuple[str, ...] = tuple(['ال', 'وال', 'بال', 'كال', 'فال'])

DUPLICATE_ROOTS: Set[str] = set(['جن', 'حد', 'حظ', 'حق', 'حل', 'خر', 'خص', 'خط', 'خل', 'خم', 'دك', 'رب', 'رد', 'سب', 'ست', 'سد', 'سر', 'سل', 'صد', 'صف', 'ضر', 'طن', 'ظل', 'عز', 'عم', 'غل', 'فخ', 'قر', 'قص', 'قل', 'مر', 'مع', 'هد', 'هم'])

FIRST_WAW_ROOTS: Set[str] = set(['أد', 'أر', 'أل', 'أم', 'بأ', 'بخ', 'بر', 'بش', 'بق', 'بل', 'به', 'تد', 'تر', 'تن', 'تي', 'ثأ', 'ثب', 'ثر', 'ثق', 'ثل', 'ثن', 'جب', 'جد', 'جر', 'جز', 'جس', 'جع', 'جف', 'جق', 'جل', 'جم', 'جن', 'جه', 'حد', 'حش', 'حف', 'حل', 'حم', 'حي', 'خز', 'خط', 'خم', 'خي', 'دج', 'دد', 'در', 'دع', 'دق', 'دك', 'دي', 'ذر', 'رب', 'رث', 'رد', 'رس', 'رش', 'رط', 'رع', 'رف', 'رق', 'رك', 'رل', 'رم', 'رن', 'ري', 'زب', 'زر', 'زز', 'زع', 'زل', 'زن', 'زي', 'سخ', 'سد', 'سط', 'سع', 'سق', 'سل', 'سم', 'سن', 'سي', 'شب', 'شج', 'شح', 'شر', 'شع', 'شق', 'شك', 'شل', 'شم', 'شن', 'شي', 'صب', 'صد', 'صف', 'صل', 'صم', 'صي', 'ضأ', 'ضب', 'ضح', 'ضر', 'ضع', 'ضم', 'طأ', 'طب', 'طد', 'طر', 'طس', 'طش', 'طف', 'طن', 'طي', 'ظب', 'ظف', 'عب', 'عث', 'عد', 'عر', 'عز', 'عس', 'عظ', 'عق', 'عك', 'عل', 'عي', 'غد', 'غر', 'غل', 'غي', 'فد', 'فر', 'فز', 'فض', 'فع', 'فق', 'في', 'قب', 'قت', 'قح', 'قد', 'قذ', 'قر', 'قص', 'قظ', 'قع', 'قف', 'قق', 'قل', 'قن', 'قي', 'كأ', 'كب', 'كد', 'كر', 'كز', 'كس', 'كع', 'كف', 'كل', 'كم', 'كن', 'كي', 'لج', 'لد', 'لس', 'لط', 'لع', 'لغ', 'لف', 'لم', 'له', 'لو', 'لي', 'مأ', 'مد', 'مس', 'مض', 'مق', 'نن', 'ني', 'هب', 'هج', 'هد', 'هر', 'هق', 'هل', 'هم', 'هن', 'هي', 'يب', 'يل'])

FIRST_YAH_ROOTS: Set[str] = set(['أس', 'بب', 'بس', 'تم', 'ثق', 'حر', 'خت', 'خن', 'رع', 'رق', 'زب', 'سر', 'سن', 'فخ', 'فع', 'قت', 'قظ', 'قن', 'مم', 'من', 'نع', 'ود', 'وم'])

LAST_ALIF_ROOTS: Set[str] = set(['أخ', 'أد', 'أز', 'بر', 'تل', 'حص', 'حي', 'خط', 'دع', 'دن', 'رخ', 'سو', 'صف', 'طر', 'عد', 'عل', 'قص', 'قو', 'لغ', 'من', 'ند'])

LAST_HAMZA_ROOTS: Set[str] = set(['بد', 'بذ', 'بر', 'بس', 'بط', 'بك', 'بو', 'تن', 'جأ', 'جر', 'جز', 'جس', 'جش', 'جف', 'حد', 'حم', 'حن', 'خب', 'خذ', 'خر', 'خس', 'خط', 'در', 'دف', 'دن', 'دو', 'ذر', 'رب', 'رج', 'رد', 'رز', 'رف', 'رق', 'زن', 'سب', 'سل', 'سو', 'سي', 'شط', 'شن', 'شي', 'صب', 'صد', 'ضو', 'طر', 'طف', 'ظم', 'عب', 'غو', 'فت', 'فث', 'فج', 'فر', 'فق', 'في', 'قث', 'قر', 'قم', 'قن', 'قي', 'كف', 'كل', 'كم', 'لب', 'لج', 'لك', 'مر', 'مل', 'من', 'مو', 'نب', 'نت', 'نس', 'نش', 'نك', 'نو', 'ني', 'هج', 'هد', 'هر', 'هز', 'هن', 'هي', 'وب', 'وث', 'وض', 'وط', 'وك', 'وم'])

LAST_MAKSOURA_ROOTS: Set[str] = set(['رأ', 'كر'])

LAST_YAH_ROOTS: Set[str] = set(['', 'أب', 'أت', 'أد', 'أذ', 'أز', 'أس', 'أق', 'أل', 'أن', 'أو', 'بد', 'بر', 'بز', 'بط', 'بغ', 'بق', 'بك', 'بل', 'بن', 'به', 'تق', 'تك', 'ثد', 'ثر', 'ثن', 'ثو', 'جب', 'جد', 'جر', 'جز', 'جل', 'جن', 'جه', 'جو', 'حد', 'حذ', 'حر', 'حش', 'حص', 'حظ', 'حف', 'حك', 'حل', 'حم', 'حن', 'حو', 'حي', 'خب', 'خز', 'خش', 'خص', 'خط', 'خف', 'خل', 'خن', 'خو', 'دخ', 'در', 'دع', 'دف', 'دل', 'دم', 'دن', 'ده', 'دو', 'ذر', 'ذك', 'ذم', 'ذو', 'رأ', 'رب', 'رث', 'رج', 'رح', 'رخ', 'رد', 'رز', 'رض', 'رع', 'رق', 'رم', 'رن', 'رو', 'ري', 'زب', 'زر', 'زك', 'زن', 'زو', 'زي', 'سب', 'سج', 'سح', 'سخ', 'سد', 'سر', 'سع', 'سف', 'سق', 'سل', 'سم', 'سو', 'شج', 'شر', 'شظ', 'شف', 'شق', 'شك', 'شه', 'شو', 'صأ', 'صب', 'صح', 'صد', 'صر', 'صغ', 'صف', 'صل', 'صم', 'ضح', 'ضن', 'ضه', 'ضو', 'طب', 'طخ', 'طر', 'طغ', 'طف', 'طق', 'طل', 'طم', 'طه', 'طو', 'ظب', 'عب', 'عت', 'عث', 'عذ', 'عر', 'عز', 'عس', 'عش', 'عص', 'عط', 'عظ', 'عل', 'عم', 'عن', 'عو', 'عي', 'غب', 'غث', 'غد', 'غر', 'غل', 'غم', 'غن', 'غو', 'غي', 'فت', 'فد', 'فر', 'فص', 'فض', 'فع', 'فل', 'فن', 'قد', 'قذ', 'قر', 'قص', 'قض', 'قع', 'قل', 'قن', 'قه', 'قو', 'قي', 'كب', 'كخ', 'كد', 'كر', 'كس', 'كف', 'كل', 'كم', 'كن', 'كه', 'كو', 'كي', 'لب', 'لح', 'لذ', 'لش', 'لظ', 'لق', 'له', 'لو', 'لي', 'مح', 'مد', 'مر', 'مز', 'مس', 'مش', 'مض', 'مط', 'مع', 'مل', 'من', 'مه', 'نأ', 'ند', 'نس', 'نع', 'نغ', 'نف', 'نق', 'نك', 'نم', 'نه', 'نو', 'ني', 'هد', 'هذ', 'هر', 'هن', 'هو', 'وت', 'وح', 'وخ', 'ود', 'ور', 'وز', 'وس', 'وش', 'وص', 'وط', 'وع', 'وغ', 'وف', 'وق', 'وك', 'ول', 'ون', 'وه'])

MID_WAW_ROOTS: Set[str] = set(['أب', 'أج', 'أد', 'أر', 'أز', 'أش', 'أض', 'أف', 'أق', 'أل', 'أن', 'أه', 'أي', 'بب', 'بت', 'بح', 'بخ', 'بر', 'بز', 'بس', 'بش', 'بص', 'بغ', 'بق', 'بل', 'تب', 'تج', 'تق', 'ته', 'ثب', 'ثر', 'ثل', 'ثي', 'جب', 'جح', 'جد', 'جر', 'جز', 'جس', 'جط', 'جع', 'جف', 'جق', 'جل', 'جن', 'جو', 'جي', 'حب', 'حت', 'حث', 'حج', 'حد', 'حذ', 'حر', 'حز', 'حس', 'حش', 'حص', 'حض', 'حط', 'حف', 'حق', 'حك', 'حل', 'حم', 'حي', 'خج', 'خخ', 'خذ', 'خر', 'خص', 'خض', 'خف', 'خل', 'خن', 'خو', 'خي', 'دأ', 'دب', 'دح', 'دخ', 'دد', 'در', 'دس', 'دش', 'دط', 'دغ', 'دف', 'دق', 'دك', 'دل', 'دم', 'دن', 'دي', 'ذب', 'ذد', 'ذق', 'ذي', 'رب', 'رث', 'رج', 'رح', 'رد', 'رز', 'رس', 'رض', 'رع', 'رغ', 'رق', 'رل', 'رم', 'ري', 'زج', 'زح', 'زد', 'زر', 'زغ', 'زق', 'زل', 'زم', 'زن', 'زي', 'سأ', 'سج', 'سح', 'سخ', 'سد', 'سر', 'سس', 'سط', 'سع', 'سغ', 'سف', 'سق', 'سك', 'سل', 'سم', 'سي', 'شب', 'شح', 'شر', 'شش', 'شط', 'شظ', 'شف', 'شق', 'شك', 'شل', 'شم', 'شن', 'شه', 'شي', 'صب', 'صت', 'صج', 'صح', 'صخ', 'صر', 'صص', 'صع', 'صغ', 'صف', 'صل', 'صم', 'صن', 'ضأ', 'ضج', 'ضر', 'ضع', 'ضي', 'طب', 'طح', 'طد', 'طر', 'طس', 'طش', 'طع', 'طف', 'طق', 'طل', 'طي', 'عج', 'عد', 'عذ', 'عر', 'عز', 'عص', 'عض', 'عف', 'عق', 'عل', 'عم', 'عن', 'عه', 'عي', 'غأ', 'غث', 'غر', 'غز', 'غش', 'غص', 'غط', 'غغ', 'غل', 'غي', 'فت', 'فج', 'فح', 'فد', 'فر', 'فز', 'فض', 'فط', 'فع', 'فف', 'فق', 'فل', 'فه', 'قب', 'قت', 'قح', 'قد', 'قر', 'قس', 'قش', 'قص', 'قض', 'قط', 'قع', 'قق', 'قل', 'قم', 'قن', 'قه', 'قي', 'كب', 'كخ', 'كد', 'كر', 'كز', 'كس', 'كش', 'كع', 'كف', 'كك', 'كم', 'كن', 'كي', 'لب', 'لث', 'لج', 'لح', 'لذ', 'لز', 'لس', 'لص', 'لط', 'لع', 'لف', 'لق', 'لك', 'لم', 'لن', 'لي', 'مأ', 'مت', 'مج', 'مر', 'مل', 'من', 'مه', 'نأ', 'نب', 'نت', 'نح', 'نخ', 'ند', 'نر', 'نس', 'نش', 'نص', 'نط', 'نع', 'نف', 'نق', 'نل', 'نم', 'نن', 'نه', 'نو', 'ني', 'هب', 'هت', 'هج', 'هد', 'هر', 'هس', 'هش', 'هع', 'هل', 'هم', 'هن', 'هو', 'هي', 'يد', 'يم'])

MID_YAH_ROOTS: Set[str] = set(['أب', 'أد', 'أر', 'أس', 'أض', 'أك', 'أل', 'أم', 'أن', 'بت', 'بد', 'بض', 'بع', 'بن', 'تح', 'تر', 'تس', 'تل', 'تم', 'ته', 'جأ', 'جب', 'جر', 'جش', 'جف', 'جل', 'حث', 'حد', 'حر', 'حز', 'حص', 'حض', 'حط', 'حف', 'حق', 'حك', 'حل', 'حن', 'حو', 'حي', 'خب', 'خر', 'خس', 'خش', 'خط', 'خل', 'خم', 'دث', 'در', 'دس', 'دك', 'دم', 'دن', 'ذع', 'ذل', 'رب', 'رث', 'رح', 'رس', 'رش', 'رض', 'رع', 'رف', 'رق', 'رل', 'رم', 'رن', 'ري', 'زت', 'زح', 'زد', 'زر', 'زز', 'زغ', 'زف', 'زق', 'زل', 'زن', 'زي', 'سأ', 'سب', 'سج', 'سح', 'سخ', 'سد', 'سر', 'سس', 'سف', 'سل', 'شأ', 'شب', 'شت', 'شح', 'شخ', 'شد', 'شط', 'شع', 'شف', 'شق', 'شل', 'شم', 'شن', 'صب', 'صح', 'صد', 'صر', 'صص', 'صع', 'صغ', 'صف', 'صم', 'صن', 'ضر', 'ضع', 'ضف', 'ضق', 'ضم', 'طب', 'طح', 'طر', 'طش', 'طع', 'طف', 'طن', 'عب', 'عث', 'عد', 'عر', 'عس', 'عش', 'عط', 'عف', 'عل', 'عن', 'عه', 'عي', 'غب', 'غث', 'غد', 'غر', 'غض', 'غط', 'غظ', 'غل', 'غم', 'غن', 'غي', 'فأ', 'فح', 'فد', 'فش', 'فض', 'فظ', 'فف', 'فل', 'فن', 'قأ', 'قح', 'قد', 'قر', 'قس', 'قش', 'قض', 'قظ', 'قف', 'قل', 'قم', 'قن', 'قي', 'كد', 'كر', 'كس', 'كف', 'كل', 'كن', 'كي', 'لث', 'لس', 'لف', 'لق', 'لل', 'لم', 'لن', 'لي', 'مت', 'مح', 'مد', 'مر', 'مز', 'مس', 'مط', 'مع', 'مل', 'من', 'نأ', 'نب', 'نح', 'نر', 'نع', 'نف', 'نق', 'نك', 'نل', 'نم', 'ني', 'هأ', 'هب', 'هت', 'هج', 'هر', 'هش', 'هض', 'هط', 'هف', 'هل', 'هم', 'هن', 'وب', 'ول'])

KHOJA_PREFIXES: Tuple[str, ...] = tuple(['لل', 'ل', 'ا', 'و', 'س', 'ب', 'ي', 'ن', 'م', 'ت', 'ف'])

KHOJA_SUFFIXES: Tuple[str, ...] = tuple(['هما', 'تما', 'كما', 'ان', 'ها', 'وا', 'تم', 'كم', 'تن', 'كن', 'نا', 'تا', 'ما', 'ون', 'ين', 'هن', 'هم', 'ته', 'تي', 'ني', 'ن', 'ك', 'ه', 'ة', 'ت', 'ا', 'ي', 'ات'])

KHOJA_PATTERNS: Tuple[str, ...] = tuple(['فعّل', 'فاعل', 'افعل', 'تفعّل', 'تفعل', 'تفاعل', 'انفعل', 'افتعل', 'افعلّ', 'استفعل', 'تفعيل', 'فعال', 'افعال', 'تفعل', 'تفاعل', 'انفعال', 'افتعال', 'افعلال', 'استفعال', 'مفعل', 'مفاعل', 'مفعل', 'متفعّل', 'متفعل', 'متفاعل', 'منفعل', 'مفتعل', 'مفعلّ', 'مستفعل', 'مفعول', 'فعول', 'مفعال', 'فعّال', 'فعيل', 'أفعل', 'فعلان', 'فعلاء', 'فعلى', 'فواعل', 'مفاعيل', 'افاعل', 'فعيّل', 'يفتعل', 'يستفعل', 'تفتعل', 'فعائل'])

STRANGE_WORDS: Set[str] = set(['إيران', 'الأوروبية', 'الفرنسي', 'بلورة', 'تحتلها', 'تلفزة', 'خوجة', 'ديسمبر', 'شيراك', 'فرنسا', 'كوفي', 'مانديلا', 'متلفزة', 'نيلسون', 'وكوفي'])

KHOJA_STOPWORDS: Set[str] = set(['،', '؟', 'ء', 'ءَ', 'آ', 'آب', 'آذار', 'آض', 'آل', 'آمينَ', 'آناء', 'آنفا', 'آه', 'آهاً', 'آهٍ', 'آهِ', 'آي', 'أ', 'أبدا', 'أبريل', 'أبو', 'أجل', 'أجمع', 'أحد', 'أخبر', 'أخذ', 'أخو', 'أربع', 'أربعاء', 'أربعة', 'أربعمئة', 'أربعمائة', 'أرى', 'أسكن', 'أصبح', 'أصلا', 'أضحى', 'أطعم', 'أعطى', 'أعلم', 'أغسطس', 'أفريل', 'أفٍّ', 'أقبل', 'أكتوبر', 'أل', 'ألا', 'ألف', 'ألفى', 'أم', 'أما', 'أمام', 'أمامك', 'أمامكَ', 'أمد', 'أمس', 'أمسى', 'أمّا', 'أن', 'أنا', 'أنبأ', 'أنت', 'أنتم', 'أنتما', 'أنتن', 'أنتِ', 'أنشأ', 'أنه', 'أنّى', 'أنًّ', 'أهلا', 'أو', 'أوت', 'أوشك', 'أول', 'أولئك', 'أولاء', 'أولالك', 'أوّهْ', 'أى', 'أي', 'أيا', 'أيار', 'أيضا', 'أيلول', 'أين', 'أيّ', 'أيّان', 'أُفٍّ', 'ؤ', 'إحدى', 'إذ', 'إذا', 'إذاً', 'إذما', 'إذن', 'إزاء', 'إلى', 'إلي', 'إليكم', 'إليكما', 'إليكنّ', 'إليكَ', 'إليه', 'إلَيْكَ', 'إلّا', 'إما', 'إمّا', 'إن', 'إنّما', 'إنَّ', 'إى', 'إياك', 'إياكم', 'إياكما', 'إياكن', 'إيانا', 'إياه', 'إياها', 'إياهم', 'إياهما', 'إياهن', 'إياي', 'إيهٍ', 'ئ', 'ا?', 'ابتدأ', 'اتخذ', 'اثنا', 'اثنان', 'اثني', 'اثنين', 'اخلولق', 'اذ', 'اذا', 'اربعون', 'اربعين', 'ارتدّ', 'استحال', 'اصبح', 'اضحى', 'الآن', 'الألاء', 'الألى', 'الا', 'التي', 'الذى', 'الذي', 'الذين', 'اللائي', 'اللاتي', 'اللاواتي', 'اللتان', 'اللتي', 'اللتيا', 'اللتين', 'اللذان', 'اللذي', 'اللذين', 'اللواتي', 'الى', 'الي', 'اليه', 'اليها', 'اليوم', 'اما', 'امسى', 'ان', 'انبرى', 'انقلب', 'انه', 'او', 'اول', 'اي', 'ب', 'بؤسا', 'بإن', 'بئس', 'باء', 'بات', 'بخٍ', 'بد', 'بدلا', 'بسّ', 'بضع', 'بطآن', 'بعد', 'بعدا', 'بعض', 'بغتة', 'بك', 'بكم', 'بكما', 'بكن', 'بل', 'بلى', 'بما', 'بماذا', 'بمن', 'بنا', 'به', 'بها', 'بهذا', 'بهم', 'بهما', 'بهن', 'بي', 'بيد', 'بين', 'بَسْ', 'بَلْهَ', 'ة', 'ت', 'تاء', 'تارة', 'تاسع', 'تانِ', 'تانِك', 'تبدّل', 'تجاه', 'تحت', 'تحوّل', 'تخذ', 'ترك', 'تسع', 'تسعة', 'تسعمئة', 'تسعمائة', 'تسعون', 'تسعين', 'تشرين', 'تعسا', 'تعلَّم', 'تكون', 'تلقاء', 'تلك', 'تلكما', 'تموز', 'تينك', 'تَيْنِ', 'تِه', 'تِي', 'ث', 'ثاء', 'ثالث', 'ثامن', 'ثان', 'ثاني', 'ثلاث', 'ثلاثاء', 'ثلاثة', 'ثلاثمئة', 'ثلاثمائة', 'ثلاثون', 'ثلاثين', 'ثم', 'ثمان', 'ثمانمئة', 'ثمانون', 'ثماني', 'ثمانية', 'ثمانين', 'ثمنمئة', 'ثمّ', 'ثمّة', 'ثمَّ', 'ج', 'جانفي', 'جدا', 'جعل', 'جلل', 'جمعة', 'جميع', 'جنيه', 'جوان', 'جويلية', 'جير', 'جيم', 'ح', 'حاء', 'حادي', 'حار', 'حاشا', 'حاي', 'حبذا', 'حتى', 'حجا', 'حدَث', 'حرى', 'حزيران', 'حسب', 'حقا', 'حمدا', 'حمو', 'حول', 'حيث', 'حيثما', 'حين', 'حيَّ', 'حَذارِ', 'خ', 'خاء', 'خاصة', 'خال', 'خامس', 'خبَّر', 'خلا', 'خلافا', 'خلف', 'خمس', 'خمسة', 'خمسمئة', 'خمسمائة', 'خمسون', 'خمسين', 'خميس', 'د', 'دال', 'درهم', 'درى', 'دواليك', 'دولار', 'دون', 'دونك', 'ديسمبر', 'دينار', 'ذ', 'ذا', 'ذات', 'ذاك', 'ذال', 'ذانك', 'ذانِ', 'ذلك', 'ذلكم', 'ذلكما', 'ذلكن', 'ذهب', 'ذو', 'ذوا', 'ذواتا', 'ذواتي', 'ذيت', 'ذينك', 'ذَيْنِ', 'ذِه', 'ذِي', 'ر', 'رأى', 'راء', 'رابع', 'راح', 'رجع', 'رزق', 'رويدك', 'ريال', 'ريث', 'رُبَّ', 'ز', 'زاي', 'زعم', 'زود', 'س', 'ساء', 'ساءما', 'سابع', 'سادس', 'سبت', 'سبتمبر', 'سبحان', 'سبع', 'سبعة', 'سبعمئة', 'سبعمائة', 'سبعون', 'سبعين', 'ست', 'ستة', 'ستكون', 'ستمئة', 'ستمائة', 'ستون', 'ستين', 'سحقا', 'سرا', 'سرعان', 'سقى', 'سمعا', 'سنتيم', 'سوف', 'سوى', 'سين', 'ش', 'شباط', 'شبه', 'شرع', 'شمال', 'شيكل', 'شين', 'شَتَّانَ', 'ص', 'صاد', 'صار', 'صباح', 'صبر', 'صبرا', 'صدقا', 'صراحة', 'صهٍ', 'صهْ', 'ض', 'ضاد', 'ضحوة', 'ضد', 'ضمن', 'ط', 'طاء', 'طاق', 'طالما', 'طرا', 'طفق', 'طَق', 'ظ', 'ظاء', 'ظل', 'ظلّ', 'ظنَّ', 'ع', 'عاد', 'عاشر', 'عامة', 'عجبا', 'عدا', 'عدَّ', 'عسى', 'عشر', 'عشرة', 'عشرون', 'عشرين', 'عل', 'علق', 'علم', 'على', 'علي', 'عليك', 'عليه', 'عليها', 'علًّ', 'عما', 'عن', 'عند', 'عنه', 'عنها', 'عوض', 'عيانا', 'عين', 'عَدَسْ', 'غ', 'غادر', 'غالبا', 'غدا', 'غداة', 'غير', 'غين', 'ف', 'فإن', 'فاء', 'فانه', 'فبراير', 'فرادى', 'فضلا', 'فقد', 'فقط', 'فكان', 'فلان', 'فلس', 'فمنذ', 'فمنها', 'فمنهم', 'فهو', 'فو', 'فوق', 'في', 'فيفري', 'فيه', 'فيها', 'ق', 'قاطبة', 'قاف', 'قام', 'قبل', 'قد', 'قرش', 'قطّ', 'قلما', 'ك', 'كأن', 'كأنّ', 'كأين', 'كأيّ', 'كأيّن', 'كاد', 'كاف', 'كان', 'كانت', 'كانون', 'كثيرا', 'كذا', 'كذلك', 'كرب', 'كسا', 'كل', 'كلا', 'كلاكما', 'كلاهما', 'كلتا', 'كلها', 'كلّما', 'كلَّا', 'كم', 'كما', 'كي', 'كيت', 'كيف', 'كيفما', 'كِخ', 'ل', 'لأن', 'لئن', 'لا', 'لا سيما', 'لات', 'لازال', 'لاسيما', 'لام', 'لايزال', 'لبيك', 'لدن', 'لدى', 'لدي', 'لذلك', 'لعل', 'لعلَّ', 'لعمر', 'لقد', 'لك', 'لكم', 'لكما', 'لكن', 'لكنه', 'لكنَّ', 'لكي', 'لكيلا', 'لم', 'لما', 'لمّا', 'لن', 'لنا', 'له', 'لها', 'لهذا', 'لهم', 'لهما', 'لهن', 'لو', 'لولا', 'لوما', 'لي', 'ليت', 'ليرة', 'ليس', 'ليسب', 'م', 'مئة', 'مئتان', 'مئتين', 'ما', 'ما برح', 'مائة', 'ماانفك', 'مابرح', 'مادام', 'ماذا', 'مارس', 'مازال', 'مافتئ', 'ماي', 'مايزال', 'مايو', 'متى', 'مثل', 'مذ', 'مرّة', 'مساء', 'مع', 'معاذ', 'معه', 'مكانكم', 'مكانكما', 'مكانكنّ', 'مكانَك', 'مليم', 'مما', 'ممن', 'من', 'منذ', 'منه', 'منها', 'مه', 'مهما', 'ميم', 'ن', 'نبَّا', 'نحن', 'نحو', 'نعم', 'نفس', 'نوفمبر', 'نون', 'نيسان', 'نيف', 'نَخْ', 'نِعِمّا', 'ه', 'هؤلاء', 'ها', 'هاء', 'هاؤم', 'هاكَ', 'هاهنا', 'هبّ', 'هذا', 'هذه', 'هكذا', 'هل', 'هللة', 'هلم', 'هلّا', 'هم', 'هما', 'همزة', 'هن', 'هنا', 'هناك', 'هنالك', 'هو', 'هي', 'هيا', 'هيت', 'هيهات', 'هيّا', 'هَؤلاء', 'هَاتانِ', 'هَاتَيْنِ', 'هَاتِه', 'هَاتِي', 'هَجْ', 'هَذا', 'هَذانِ', 'هَذَيْنِ', 'هَذِه', 'هَذِي', 'هَيْهات', 'و', 'وأبو', 'وأن', 'وا', 'واحد', 'والتي', 'والذي', 'وان', 'واهاً', 'واو', 'وبه', 'وبين', 'وتلك', 'وثي', 'وجد', 'وحتى', 'وذلك', 'وراءَك', 'ورد', 'وعلى', 'وعن', 'وفي', 'وقد', 'وكان', 'وكانت', 'وكذلك', 'ولايزال الحالي', 'ولكن', 'وله', 'وليس', 'ومع', 'ومن', 'ومنذ', 'وهب', 'وهذا', 'وهذه', 'وهو', 'وهي', 'ويكأنّ', 'ويكون', 'وَيْ', 'وُشْكَانَ', 'ى', 'ي', 'ياء', 'يكون', 'يلي', 'يمكن', 'يمين', 'ين', 'يناير', 'يوان', 'يورو', 'يوليو', 'يوم', 'يومئذ', 'يونيو'])

# Arabic short vowels / diacritics to strip
DIACRITICS = set('ًٌٍَُِّْـ')

# Non-letter / punctuation characters
PUNCTUATIONS = set('.,:;!?¿¡«»"\'()[]{}<>/*-+_=|#@$%^&~`')


def load_canonical_roots(dict_path: Optional[str] = None) -> Set[str]:
    """
    Load the canonical 4,748 Arabic roots lexicon.

    Attempts to load from:
    1. Specified `dict_path` if provided and exists.
    2. Standard repository paths: `data/khoja_roots.txt` or adjacent directories.
    3. Embedded base64-gzipped fallback lexicon (zero-dependency guarantee).
    """
    candidate_paths = []
    if dict_path:
        candidate_paths.append(dict_path)

    # Standard project paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_paths.extend([
        os.path.join(base_dir, '..', '..', '..', 'data', 'khoja_roots.txt'),
        os.path.join(base_dir, '..', '..', 'data', 'khoja_roots.txt'),
        os.path.join(base_dir, 'khoja_roots.txt'),
        os.path.join('data', 'khoja_roots.txt'),
    ])

    for p in candidate_paths:
        if os.path.isfile(p):
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    roots = set(line.strip() for line in f if line.strip())
                if len(roots) >= 4700:
                    return roots
            except Exception:
                pass

    # Fallback to embedded base64-gzipped dictionary
    try:
        raw = gzip.decompress(base64.b64decode(EMBEDDED_KHOJA_ROOTS_B64.encode('ascii')))
        roots = set(raw.decode('utf-8').splitlines())
        return roots
    except Exception as e:
        raise RuntimeError(f"Failed to load Khoja root dictionary: {e}")


class KhojaStemmer:
    """
    Khoja dictionary-based morphological root extractor.
    Conforms strictly to the IR benchmark interface contract: stem_text(text: str) -> str.
    """

    def __init__(self, dict_path: Optional[str] = None):
        self.roots = load_canonical_roots(dict_path)
        # Separate into triliteral and quadriliteral sets for fast membership testing
        self.tri_roots = set(r for r in self.roots if len(r) == 3)
        self.quad_roots = set(r for r in self.roots if len(r) == 4)

    def stem_token(self, token: str) -> str:
        """
        Stem a single Arabic token to its morphological root.

        Returns root if successfully extracted and verified in dictionary;
        otherwise falls back to light-stemmed or original token.
        """
        if not token:
            return ""

        # Non-Arabic tokens returned verbatim
        if not any("\u0600" <= c <= "\u06FF" for c in token):
            return token

        # Strip diacritics
        clean = "".join(c for c in token if c not in DIACRITICS)
        # Strip punctuation
        clean = "".join(c for c in clean if c not in PUNCTUATIONS)
        # Keep alphabetic Arabic characters
        word = "".join(c for c in clean if c.isalpha())
        if not word:
            return token

        # If foreign / strange word or stopword, preserve verbatim
        if word in STRANGE_WORDS or word in KHOJA_STOPWORDS:
            return word

        flags = [False, False, False, False, False, False]
        stemmed = self._stem_word(word, flags)

        # If root found and verified, return it
        if flags[0]:
            return stemmed

        # Fallback: if not found, return cleaned word or original
        return stemmed if stemmed else token

    def _stem_word(self, word: str, flags: List[bool]) -> str:
        if len(word) == 2:
            word = self._is_two_letters(word, flags)
        if len(word) == 3 and not flags[0]:
            word = self._is_three_letters(word, flags)
        if len(word) == 4:
            self._is_four_letters(word, flags)

        if not flags[0]:
            word = self._check_patterns(word, flags)
        if not flags[0]:
            word = self._check_definite_article(word, flags)
        if not flags[0] and not flags[2]:
            word = self._check_prefix_waw(word, flags)
        if not flags[0] and not flags[2]:
            word = self._check_for_suffixes(word, flags)
        if not flags[0] and not flags[2]:
            word = self._check_for_prefixes(word, flags)

        return word

    def _is_two_letters(self, word: str, flags: List[bool]) -> str:
        word = self._duplicate(word, flags)
        if not flags[0]:
            word = self._last_weak(word, flags)
        if not flags[0]:
            word = self._first_weak(word, flags)
        if not flags[0]:
            word = self._middle_weak(word, flags)
        return word

    def _duplicate(self, word: str, flags: List[bool]) -> str:
        if word in DUPLICATE_ROOTS:
            flags[0] = True
            return word + word[1]
        return word

    def _last_weak(self, word: str, flags: List[bool]) -> str:
        if word in LAST_ALIF_ROOTS:
            flags[0] = True
            return word + "ا"
        elif word in LAST_HAMZA_ROOTS:
            flags[0] = True
            return word + "أ"
        elif word in LAST_MAKSOURA_ROOTS:
            flags[0] = True
            return word + "ى"
        elif word in LAST_YAH_ROOTS:
            flags[0] = True
            return word + "ي"
        return word

    def _first_weak(self, word: str, flags: List[bool]) -> str:
        if word in FIRST_WAW_ROOTS:
            flags[0] = True
            return "و" + word
        elif word in FIRST_YAH_ROOTS:
            flags[0] = True
            return "ي" + word
        return word

    def _middle_weak(self, word: str, flags: List[bool]) -> str:
        if word in MID_WAW_ROOTS:
            flags[0] = True
            return word[0] + "و" + word[1:]
        elif word in MID_YAH_ROOTS:
            flags[0] = True
            return word[0] + "ي" + word[1:]
        return word

    def _is_three_letters(self, word: str, flags: List[bool]) -> str:
        root = ""
        if len(word) > 0:
            if word[0] in ("ا", "ؤ", "ئ"):
                root = "أ" + word[1:]

            if len(word) >= 3:
                # Weak final radical
                if word[2] in ("و", "ي", "ا", "ى", "ء", "ئ"):
                    r = self._last_weak(word[:2], flags)
                    if flags[0]:
                        return r

                # Weak medial radical
                if word[1] in ("و", "ي", "ا", "ئ"):
                    r = self._middle_weak(word[0] + word[2], flags)
                    if flags[0]:
                        return r

                # Hamza medial radical
                if word[1] in ("ؤ", "ئ"):
                    if word[2] in ("م", "ز", "ر"):
                        root = word[0] + "ا" + word[2]
                    else:
                        root = word[0] + "أ" + word[2]

                # Shaddah gemination
                if word[2] == "ّ":
                    root = word[0] + word[1] + word[1]

        if not root:
            if word in self.tri_roots:
                flags[0] = True
                return word
        elif root in self.tri_roots:
            flags[0] = True
            return root

        return word

    def _is_four_letters(self, word: str, flags: List[bool]) -> None:
        if word in self.quad_roots:
            flags[0] = True

    def _check_patterns(self, word: str, flags: List[bool]) -> str:
        if len(word) > 0 and word[0] in ("أ", "إ", "آ"):
            word = "ا" + word[1:]

        for pattern in KHOJA_PATTERNS:
            if len(pattern) == len(word):
                num_same = 0
                for j in range(len(word)):
                    if pattern[j] == word[j] and pattern[j] not in ("ف", "ع", "ل"):
                        num_same += 1

                # Form ift'ala / if'allal check
                if len(word) == 6 and word[3] == word[5] and num_same == 2:
                    m_word = word[1] + word[2] + word[3]
                    m_word = self._is_three_letters(m_word, flags)
                    if flags[0]:
                        return m_word

                if len(word) - 3 <= num_same:
                    root_chars = []
                    for j in range(len(word)):
                        if pattern[j] in ("ف", "ع", "ل"):
                            root_chars.append(word[j])
                    m_word = "".join(root_chars)
                    m_word = self._is_three_letters(m_word, flags)
                    if flags[0]:
                        return m_word

        return word

    def _check_definite_article(self, word: str, flags: List[bool]) -> str:
        m_word = ""
        for da in DEFINITE_ARTICLES:
            if word.startswith(da):
                m_word = word[len(da):]
                if m_word in KHOJA_STOPWORDS:
                    return m_word

                if len(m_word) == 2:
                    m_word = self._is_two_letters(m_word, flags)
                elif len(m_word) == 3 and not flags[0]:
                    m_word = self._is_three_letters(m_word, flags)
                elif len(m_word) == 4:
                    self._is_four_letters(m_word, flags)

                if not flags[0] and len(m_word) > 2:
                    m_word = self._check_patterns(m_word, flags)
                if not flags[0] and not flags[2]:
                    m_word = self._check_for_suffixes(m_word, flags)
                if not flags[0] and not flags[2]:
                    m_word = self._check_for_prefixes(m_word, flags)

                if flags[2]:
                    return m_word
                if flags[0] and not flags[2]:
                    return m_word

        if len(m_word) > 3:
            return m_word
        return word

    def _check_prefix_waw(self, word: str, flags: List[bool]) -> str:
        if len(word) > 3 and word.startswith("و"):
            m_word = word[1:]
            if m_word in KHOJA_STOPWORDS:
                return m_word

            if len(m_word) == 2:
                m_word = self._is_two_letters(m_word, flags)
            elif len(m_word) == 3 and not flags[0]:
                m_word = self._is_three_letters(m_word, flags)
            elif len(m_word) == 4:
                self._is_four_letters(m_word, flags)

            if not flags[0] and len(m_word) > 2:
                m_word = self._check_patterns(m_word, flags)
            if not flags[0] and not flags[2]:
                m_word = self._check_for_suffixes(m_word, flags)
            if not flags[0] and not flags[2]:
                m_word = self._check_for_prefixes(m_word, flags)

            if flags[2]:
                return m_word
            if flags[0] and not flags[2]:
                return m_word

        return word

    def _check_for_suffixes(self, word: str, flags: List[bool]) -> str:
        m_word = word
        flags[5] = True

        for suffix in KHOJA_SUFFIXES:
            if m_word.endswith(suffix):
                m_word = m_word[:-len(suffix)]
                if m_word in KHOJA_STOPWORDS:
                    flags[5] = False
                    return m_word

                if len(m_word) == 2:
                    m_word = self._is_two_letters(m_word, flags)
                elif len(m_word) == 3:
                    m_word = self._is_three_letters(m_word, flags)
                elif len(m_word) == 4:
                    self._is_four_letters(m_word, flags)

                if not flags[0] and len(m_word) > 2:
                    m_word = self._check_patterns(m_word, flags)

                if flags[2]:
                    flags[5] = False
                    return m_word
                if flags[0]:
                    flags[5] = False
                    return m_word

        flags[5] = False
        return word

    def _check_for_prefixes(self, word: str, flags: List[bool]) -> str:
        m_word = word

        for prefix in KHOJA_PREFIXES:
            if m_word.startswith(prefix):
                m_word = m_word[len(prefix):]
                if m_word in KHOJA_STOPWORDS:
                    return m_word

                if len(m_word) == 2:
                    m_word = self._is_two_letters(m_word, flags)
                elif len(m_word) == 3 and not flags[0]:
                    m_word = self._is_three_letters(m_word, flags)
                elif len(m_word) == 4:
                    self._is_four_letters(m_word, flags)

                if not flags[0] and len(m_word) > 2:
                    m_word = self._check_patterns(m_word, flags)
                if not flags[0] and not flags[2] and not flags[5]:
                    m_word = self._check_for_suffixes(m_word, flags)

                if flags[2]:
                    return m_word
                if flags[0] and not flags[2]:
                    return m_word

        return word

    def stem_text(self, text: Optional[str]) -> str:
        """
        Stem whitespace-delimited text according to IR interface contract.

        Args:
            text: Normalized Arabic text string.

        Returns:
            Space-delimited stemmed string suitable for to_tsvector('simple', ...).
        """
        if not text:
            return ""
        tokens = text.split()
        if not tokens:
            return ""

        stemmed_tokens = [self.stem_token(t) for t in tokens]
        return " ".join(st for st in stemmed_tokens if st)

    def __call__(self, text: Optional[str]) -> str:
        return self.stem_text(text)


_DEFAULT_KHOJA = KhojaStemmer()


def stem_text(text: Optional[str]) -> str:
    """Module-level pure function adhering to interface contract."""
    return _DEFAULT_KHOJA.stem_text(text)


__all__ = [
    "KhojaStemmer",
    "load_canonical_roots",
    "stem_text",
    "DEFINITE_ARTICLES",
    "DUPLICATE_ROOTS",
    "FIRST_WAW_ROOTS",
    "FIRST_YAH_ROOTS",
    "LAST_ALIF_ROOTS",
    "LAST_HAMZA_ROOTS",
    "LAST_MAKSOURA_ROOTS",
    "LAST_YAH_ROOTS",
    "MID_WAW_ROOTS",
    "MID_YAH_ROOTS",
    "KHOJA_PREFIXES",
    "KHOJA_SUFFIXES",
    "KHOJA_PATTERNS",
    "STRANGE_WORDS",
    "KHOJA_STOPWORDS",
]
