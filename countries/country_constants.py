from collections import defaultdict
from black import err
import numpy as np
from tqdm import tqdm
from pycountry_convert import country_alpha2_to_continent_code, country_name_to_country_alpha2, country_name_to_country_alpha3, convert_country_alpha2_to_continent_code
from geopy.geocoders import Nominatim
from time import time
import pycountry

# ---------------------------------------------------------------------------------------------
#                               TRAITEMENT DES PAYS
# ---------------------------------------------------------------------------------------------

countries_possibilities = {
        "Afghanistan":["la république islamique d'afghanistan", '阿富汗', 'афганистан', 'جمهورية أفغانستان الإسلامية', 'afghanistan', '阿富汗伊斯兰共和国', 'the islamic republic of afghanistan', 'أفغانستان', 'islamic republic of afghanistan', 'la república islámica del afganistán', 'исламская республика афганистан', 'afganistán'],
        "Albania":['阿尔巴尼亚', 'республика албания', 'la república de albania', '阿尔巴尼亚共和国', 'albanie', 'the republic of albania', 'албания', 'ألبانيا', 'republic of albania', 'جمهورية ألبانيا', 'albania', "la république d'albanie"],
        "Algeria":['algerie', 'algérie',"people's democratic algeria",'алжирская народная демократическая республика', 'la république algérienne démocratique et populaire', '阿尔及利亚', "the people's democratic republic of algeria", 'la república argelina democrática y popular', 'الجمهورية الجزائرية الديمقراطية الشعبية', 'argelia', 'algeria', "people's democratic republic of algeria", '阿尔及利亚民主人民共和国', 'algérie', 'الجزائر', 'алжир', "people's democratic algeria"],
        "Angola":['angola', 'la república de angola', '安哥拉', 'republic of angola', '安哥拉共和国', "la république d'angola", 'أنغولا', 'республика ангола', 'the republic of angola', 'جمهورية أنغولا', 'ангола'],
        "Antigua and Barbuda":['antigua-et-barbuda', 'антигуа и барбуда', 'antigua y barbuda', 'antigua and barbuda', '安提瓜和巴布达', 'أنتيغوا وبربودا'],
        "Argentina":['argentine', 'Argentine', '阿根廷', 'the argentine republic', 'аргентина', 'الأرجنتين', 'جمهورية الأرجنتين', 'аргентинская республика', 'la república argentina', 'argentina', 'argentine republic', 'la république argentine', '阿根廷共和国', 'argentine'],
        "Armenia":['arménie', 'armenia', '亚美尼亚', 'республика армения', 'republic of armenia', "la république d'arménie", 'la república de armenia', 'أرمينيا', 'جمهورية أرمينيا', 'армения', 'the republic of armenia', '亚美尼亚共和国'],
        "Australia":['австралия', 'أستراليا', 'australia', '澳大利亚', "l'australie", 'australie', 'null-australia'],
        "Austria":['republic of austria', 'austria', 'جمهورية النمسا', 'the republic of austria', 'la república de austria', '奥地利共和国', 'autriche', 'австрийская республика', 'النمسا', "la république d'autriche", 'австрия', '奥地利'],
        "Azerbaijan":['republic of azerbaijan', 'جمهورية أذربيجان', 'the republic of azerbaijan', '阿塞拜疆共和国', 'azerbaiyán', 'azerbaijan', 'азербайджан', 'أذربيجان', 'азербайджанская республика', '阿塞拜疆', 'la república de azerbaiyán', 'azerbaïdjan', "la république d'azerbaïdjan"],
        "Bahamas":['the-bahamas', "commonwealth of the bahamas", 'el commonwealth de las bahamas', 'the commonwealth of the bahamas', 'le commonwealth des bahamas', '巴哈马 国', 'bahamas', 'commonwealth of the bahamas', 'содружество багамских островов', 'جزر البهاما', 'багамские острова', '巴哈马', 'كمنولث جزر البهاما'],
        "Bahrain":['巴林王国', 'البحرين', 'bahrain', 'the kingdom of bahrain', 'le royaume de bahreïn', 'kingdom of bahrain', 'مملكة البحرين', 'bahreïn', 'бахрейн', '巴林', 'bahrein', 'королевство бахрейн', 'el reino de bahrein'],
        "Barbados":['барбадос', 'barbados', 'بربادوس', '巴巴多斯', 'la barbade', 'barbade'],
        "Belarus":['беларусь', 'بيلاروس', 'the republic of belarus', 'la république du bélarus', 'republic of belarus', '白俄罗斯共和国', 'biélorussie', 'bélarus', '白俄罗斯', 'la república de belarús', 'республика беларусь', 'belarús', 'جمهورية بيلاروس', 'belarus'],
        "Belgium":['королевство бельгия', 'bélgica', 'kingdom of belgium', 'مملكة بلجيكا', 'belgium', 'the kingdom of belgium', 'belgica', '比利时', 'بلجيكا', 'belgique', 'belgie', 'le royaume de belgique', 'бельгия', 'el reino de bélgica', '比利时王国', 'belgio', 'belgien'],
        "Belize":['بليز', 'le belize', 'белиз', '伯利兹', 'belize', 'belice'],
        "Benin":['la république du bénin', 'benin', 'бенин', 'بنن', '贝宁', 'the republic of benin', 'republic of benin', 'la república de benin', 'bénin', 'республика бенин', '贝宁共和国', 'جمهورية بنن'],
        "Bhutan":['бутан', 'bhoutan', 'le royaume du bhoutan', 'مملكة بوتان', 'bhutan', '不丹', 'بوتان', 'bhután', '不丹王国', 'kingdom of bhutan', 'the kingdom of bhutan', 'el reino de bhután', 'королевство бутан'],
        "Bosnia And Herzegovina":['bosnia i hercegovina bosnian', 'bosnia-i-hercegovina-bosnian', 'bosnie-herzégovine', 'dominican republicvbosnia and herzegovina', "bosnia and herzegovina", 'la bosnie-herzégovine', 'bosnia and herzegovina', 'bosnie-herzégovine', '波斯尼亚和黑塞哥维那', 'bosnia y herzegovina', 'البوسنة والهرسك', 'republic of bosnia and herzegovina', 'босния и герцеговина'],
        "Botswana":['республика ботсвана', 'republic of botswana', 'the republic of botswana', 'la república de botswana', 'بوتسوانا', 'ботсвана', '博茨瓦纳共和国', '博茨瓦纳', 'جمهورية بوتسوانا', 'botswana', 'la république du botswana'],
        "Brunei Darussalam":['بروني دار السلام', '文莱达鲁萨兰国', 'le brunéi darussalam', 'brunei darussalam', 'brunéi darussalam', 'бруней-даруссалам'],
        "Bulgaria":['республика болгария', 'болгария', 'republic of bulgaria', 'bulgarien', 'la république de bulgarie', 'la república de bulgaria', 'bulgarie', 'the republic of bulgaria', 'bulagria', 'bulgaria', '保加利亚共和国', '保加利亚', 'بلغاريا', 'جمهورية بلغاريا'],
        "Burkina Faso":['burkina faso', 'le burkina faso', 'буркина-фасо', 'بوركينا فاسو', '布基纳法索'],
        "Burundi":['the republic of burundi', 'бурунди', 'جمهورية بوروندي', 'республика бурунди', 'بوروندي', 'burundi', 'la república de burundi', '布隆迪', 'la république du burundi', '布隆迪共和国', 'republic of burundi'],
        "Cabo Verde":['佛得角共和国', 'la república de cabo verde', 'جمهورية كابو فيردي', 'кабо-верде', 'كابو فيردي', 'republic of cabo verde', '佛得角', 'cabo verde', 'республика кабо-верде', 'la république de cabo verde', 'the republic of cabo verde'],
        "Cambodia":['مملكة كمبوديا', 'камбоджа', '柬埔寨王国', 'le royaume du cambodge', 'the kingdom of cambodia', 'camboya', 'cambodia', 'kingdom of cambodia', 'el reino de camboya', 'cambodge', 'كمبوديا', 'королевство камбоджа', '柬埔寨'],
        "Cameroon":['camerún', 'cameroun', 'republic of cameroon', 'the republic of cameroon', 'الكاميرون', 'la república del camerún', '喀麦隆共和国', '喀麦隆', 'جمهورية الكاميرون', 'la république du cameroun', 'cameroon', 'республика камерун', 'камерун'],
        "Canada":['加拿大', 'canada', 'le canada', 'canadá', 'كندا', 'канада', 'el canadá'],
        "Central African Republic":['central african','república centroafricana', 'the central african republic', 'central african republic', 'central african', 'جمهورية أفريقيا الوسطى', 'центральноафриканская республика', 'la république centrafricaine', 'la república centroafricana', 'république centrafricaine', '中非共和国', 'the central african republic', 'central african republic', "Central African Republic", 'centrafrique'],
        "Chad":['tchad', 'republic of chad', '乍得', 'جمهورية تشاد', '乍得共和国', 'республика чад', 'la república del chad', 'la république du tchad', 'تشاد', 'the republic of chad', 'chad', 'чад'],
        "Chile":['chile', '智利共和国', 'республика чили', '智利', 'la république du chili', 'republic of chile', 'جمهورية شيلي', 'شيلي', 'la república de chile', 'чили', 'the republic of chile', 'chili'],
        "China":['chine'],
        "Colombia":['la república de colombia', 'كولومبيا', 'colombie', 'the republic of colombia', '哥伦比亚共和国', '哥伦比亚', 'колумбия', 'la république de colombie', 'جمهورية كولومبيا', 'colombia', 'республика колумбия', 'republic of colombia'],
        "Commonwealth of Dominica":['多米尼克国', 'dominica', '多米尼克', 'содружество доминики', 'دومينيكا', 'dominique', 'el commonwealth de dominica', 'доминика', 'le commonwealth de dominique', 'commonwealth of dominica', 'the commonwealth of dominica', 'كمنولث دومينيكا'],
        "Cook Islands":['las islas cook', 'the cook islands', '库克群岛', 'جزر كوك', 'les îles cook', 'îles cook', 'cook islands', 'islas cook', 'острова кука'],
        "Congo, The Democratic Republic of the":['Democratic Republic Of The Congo','democratic republic Of the congo','congo (kinshasa)', 'kinshasa', '刚果共和国', 'جمهورية الكونغو', 'конго',  'congo (rdc)','جمهورية الكونغو الديمقراطية', 'république démocratique du congo', 'república democrática del congo','the democratic republic of the congo', 'la república democrática del congo', '刚果民主共和国', 'демократическая республика конго','la république démocratique du congo', 'democratic republic of the congo','democratic congo','democratic-democratic-republic-of-the-congo', 'democratic-republic-of-the-congo'],
        "Republic of the Congo" : ['congo (brazzaville)','brazzaville',"the republic of the congo", 'republic of the congo', '刚果（布）', 'республика конго','the republic of the congo', 'la république du congo', 'congo', 'la república del congo', 'republic-of-the-congo',  'الكونغو'],
        "Costa Rica":['коста-рика', 'كوستاريكا', 'costa rica', 'республика коста-рика', 'republic of costa rica', 'la república de costa rica', '哥斯达黎加共和国', 'جمهورية كوستاريكا', '哥斯达黎加', 'the republic of costa rica', 'la république du costa rica'],
        "Croatia":['كرواتيا', 'la república de croacia', 'la république de croatie', '克罗地亚', 'хорватия', 'the republic of croatia', 'جمهورية كرواتيا', 'croatie', 'croatia', 'республика хорватия', 'republic of croatia', '克罗地亚共和国', 'croacia'],
        "Cuba":['كوبا', '古巴', 'the republic of cuba', 'cuba', 'la república de cuba', 'республика куба', 'куба', 'جمهورية كوبا', 'republic of cuba', '古巴共和国', 'la république de cuba'],
        "Republic of Cyprus":['chypre du nord', 'north cyprus', 'northern cyprus','chypre', 'кипр', 'κύπρο', 'cyprus', 'the republic of cyprus', 'la república de chipre', 'республика кипр', '塞浦路斯共和国', 'chipre', 'la république de chypre', '塞浦路斯', 'cyprus', 'قبرص', 'جمهورية قبرص', 'republic of cyprus'],
        "Czech Republic":['česko', 'czech-republic', 'czech-repblik', 'czechy', 'czech-republi', 'république tchèque', 'czech republic',"czech", 'czech republic', 'the czech republic', 'la république tchèque', 'чешская республика', 'الجمهورية التشيكية', 'chequia', 'czechia', '捷克', 'tchéquie', 'la república checa', 'чехия', '捷克共和国', 'تشيكيا', 'czech'],
        "Sao Tome and Principe":['sao tomé-et-principe','tomé-et-principe', 'république démocratique de sao tomé-et-principe', 'são tomé e príncipe', 'la république démocratique de sao tomé-et-principe', 'sao tome and principe','democratic sao tome and principe', 'сан-томе и принсипи', 'sao tome and principe', 'democratic republic of sao tome and principe', 'sao tomé-et-principe', '圣多美和普林西比', 'democratic sao tome and principe', 'سان تومي وبرينسيبي', 'the democratic republic of sao tome and principe', 'جمهورية سان تومي وبرينسيبي الديمقراطية', 'la república democrática de santo tomé y príncipe', 'демократическая республика сан-томе и принсипи', '圣多美和普林西比民主共和国', 'santo tomé y príncipe', 'la république démocratique de sao tomé-et-principe'],
        "Sri Lanka":["democratic socialist sri lanka",'斯里兰卡民主社会主义共和国', 'سري لانكا', 'sri lanka', 'the democratic socialist republic of sri lanka', 'democratic socialist republic of sri lanka', '斯里兰卡', 'جمهورية سري لانكا الاشتراكية الديمقراطية', 'демократическая социалистическая республика шри-ланка', 'la république socialiste démocratique de sri lanka', 'шри-ланка', 'democratic socialist sri lanka', 'la república socialista democrática de sri lanka'],
        "Timor-Leste":['democratic timor-Leste', 'timor-Leste', 'the democratic republic of timor-Leste', 'democratic republic of timor-leste', '东帝汶', 'the democratic republic of timor-leste', '东帝汶民主共和国', 'تيمور- ليشتي', 'timor-leste', 'جمهورية تيمور - ليشتي الديمقراطية', 'la república democrática de timor-leste', 'тимор-лешти', 'la république démocratique du timor-leste', 'демократическая республика тимор-лешти', 'democratic timor-leste'],
        "Denmark":['kingdom of denmark', 'الدانمرك', 'the kingdom of denmark', 'el reino de dinamarca', '丹麦王国', 'королевство дания', 'дания', 'le royaume du danemark', 'dinamarca', 'denmark', 'مملكة الدانمرك', '丹麦', 'danemark'],
        "Republic of Djibouti":['Djibouti','республика джибути', '吉布提', 'the republic of djibouti', 'djibouti', 'جمهورية جيبوتي', '吉布提共和国', 'djibouti', 'la república de djibouti', 'republic of djibouti', 'джибути', 'la république de djibouti', 'جيبوتي'],
        "Dominican Republic":['dominicaine (la république)', 'la république dominicaine','république dominicaine','dominican','доминиканская республика', '多米尼加', 'la república dominicana', 'république dominicaine', 'the dominican republic', 'la république dominicaine', 'الجمهورية الدومينيكية', 'república dominicana', 'dominican republic', '多米尼加共和国', 'dominican', "dominican republic",'republica-dominicana-espanol', 'république dominicaine'],
        "Eastern Uruguay":['uruguay', 'أوروغواي', 'جمهورية أوروغواي الشرقية', "la république orientale de l'uruguay", 'the eastern republic of uruguay', 'уругвай', 'восточная республика уругвай', '乌拉圭', 'eastern uruguay', '乌拉圭东岸共和国', 'eastern republic of uruguay', 'la república oriental del uruguay'],
        "Ecuador":["la république de l'équateur", 'republic of ecuador', 'جمهورية إكوادور', 'la república del ecuador', 'эквадор', 'ecuador', '厄瓜多尔共和国', 'республика эквадор', 'équateur', 'إكوادور', '厄瓜多尔', 'the republic of ecuador'],
        "Egypt":['egipto', "la république arabe d'égypte", 'جمهورية مصر العربية', 'египет', 'the arab republic of egypt', 'la república árabe de egipto', 'égypte', 'арабская республика египет', '埃及', 'egypt', 'arab republic of egypt', 'مصر', '阿拉伯埃及共和国'],
        "El Salvador":['сальвадор', '萨尔瓦多', 'республика эль-сальвадор', 'salvador', 'el salvador', "la république d'el salvador", 'la república de el salvador', 'السلفادور', 'republic of el salvador', 'the republic of el salvador', '萨尔瓦多共和国', 'جمهورية السلفادور'],
        "Equatorial Guinea":['la république de guinée équatoriale', 'guinea ecuatorial', '赤道几内亚', 'غينيا الاستوائية', 'جمهورية غينيا الاستوائية', 'equatorial guinea', 'экваториальная гвинея', 'republic of equatorial guinea', '赤道几内亚共和国', 'республика экваториальная гвинея', 'the republic of equatorial guinea', 'la república de guinea ecuatorial', 'guinée équatoriale'],
        "Estonia":['republic of estonia', 'إستونيا', 'the republic of estonia', 'la república de estonia', 'эстония', 'estonia', '爱沙尼亚共和国', 'جمهورية إستونيا', '爱沙尼亚', 'estonie', 'эстонская республика', "la république d'estonie"],
        "Kingdom of Eswatini":['إسواتيني', 'kingdom of eswatini','eswatini, kingdom of', 'the kingdom of eswatini', 'эсватини', 'مملكة إسواتيني', '斯威士兰', 'el reino de eswatini', 'le royaume d’eswatini', '斯威士兰王国', 'eswatini', 'королевство эсватини', 'Eswatini, Kingdom of'],
        "Ethiopia":['эфиопия', "la république fédérale démocratique d'éthiopie", 'جمهورية إثيوبيا الديمقراطية الاتحادية', 'федеративная демократическая республика эфиопия', 'etiopía', 'federal democratic republic of ethiopia', 'ethiopia', '埃塞俄比亚联邦民主共和国', 'إثيوبيا', '埃塞俄 比亚', 'the federal democratic republic of ethiopia', 'éthiopie', 'la república democrática federal de etiopía'],
        "European Union":['europa', 'europe'],
        "Brazil":['federative brazil',"Brazil","brazil",'brésil', 'brazil,pt', 'the federative republic of brazil', 'federative republic of brazil', 'republic of brazil', 'brazil','федеративная республика бразилия', 'the federative republic of brazil', 'جمهورية البرازيل الاتحادية', 'la république fédérative du brésil', 'البرازيل', '巴西联邦共和国', 'brazil', 'federative brazil', 'brésil', 'la república federativa del brasil', '巴西', 'бразилия', 'brasil', 'federative republic of brazil'],
        "Fiji":['斐济共和国', 'la république des fidji', 'la república de fiji', 'فيجي', 'republic of fiji', 'республика  фиджи', 'جمهورية فيجي', 'the republic of fiji', '斐济', 'фиджи', 'fiji', 'fidji'],
        "Finland":['финляндия', '芬兰', 'finlande', 'republic of finland', 'la república de finlandia', 'finlandia', '芬兰共和国', 'la république de finlande', 'финляндская республика', 'suomi', 'فنلندا', 'finland', 'جمهورية فنلندا', 'the republic of finland'],
        "France":['france', 'dom-tom', 'franca', 'francia', 'francja', 'frankreich', 'frankrijk', 'franța', 'paris', 'ranska',"french",'франция', '法国', 'france', 'francia', '法兰西共和国', 'французская республика', 'الجمهورية الفرنسية', 'فرنسا', 'french', 'the french republic', 'la república francesa', 'french republic', 'la république française'],
        "French Polynesia":['polinesia-francesa', 'polynesie-francaise'],
        "Gabonese":['加蓬共和国', 'la république gabonaise', 'la república gabonesa', 'gabonese', 'gabón', 'габонская республика', '加蓬', 'جمهورية الغابون', 'gabonese republic', 'gabon', 'габон', 'غابون', 'the gabonese republic'],
        "Gambia":['la república de gambia', '冈比亚', 'republic of the gambia', 'la république de gambie', 'гамбия', 'the republic of the gambia', 'республика гамбия', 'gambie', '冈比亚共和国', 'غامبيا', 'gambia', 'جمهورية غامبيا'],
        "Georgia":['la géorgie', 'georgia', 'грузия', 'géorgie', 'جورجيا', '格鲁吉亚'],
        "Germany":['germania', 'allemagne', 'la república federal de alemania', 'جمهورية ألمانيا الاتحادية', 'federal republic of germany', 'germany', 'the federal republic of germany', 'niemcy', 'deutschland', "la république fédérale d'allemagne", '德国', 'федеративная республика германия', '德意志联邦共和国', 'германия', 'east-germany', 'ألمانيا', 'alemania'],
        "Ghana":['غانا', 'гана', 'the republic of ghana', 'la république du ghana', 'la república de ghana', '加纳', '加纳共和国', 'جمهورية غانا', 'ghana', 'республика гана', 'republic of ghana'],
        "Grand Duchy of Luxembourg":['великое герцогство люксембург', 'لكسمبرغ', '卢森堡大公国', 'دوقية لكسمبرغ الكبرى', 'luxemburgo', 'el gran ducado de luxemburgo', 'the grand duchy of luxembourg', '卢森堡', 'le grand-duché de luxembourg', 'люксембург', 'luxembourg', 'grand duchy of luxembourg'],
        "Greece":['grèce', 'Hellenic', "Grèce",'greece', 'la république hellénique', 'الجمهورية الهيلينية', 'grecia', 'hellenic', 'la república helénica', 'hellenic republic', 'the hellenic republic', 'греческая республика', 'اليونان', '希腊共和国', 'греция', 'grèce', '希腊'],
        "Grenada":['格林纳达', 'غرينادا', 'granada', 'grenada', 'grenade', 'la grenade', 'гренада'],
        "Guadeloupe":['guadalupe'],
        "Guatemala":['غواتيمالا', 'la república de guatemala', '危地马拉共和国', '危地马拉', 'guatemaltecos', 'جمهورية غواتيمالا', 'гватемала', 'la république du guatemala', 'the republic of guatemala', 'guatemala', 'республика гватемала', 'republic of guatemala'],
        "Guinea":['la république de guinée', '几内亚共和国', 'republic of guinea', 'غينيا', 'la república de guinea', 'guinee', '几内亚', 'гвинейская республика', 'guinée', 'the republic of guinea', 'جمهورية غينيا', 'guinea', 'гвинея'],
        "Guinea-Bissau":['几内亚比绍共和国', 'جمهورية غينيا - بيساو', 'guinea-bissau', 'guinée-bissau', 'republic of guinea-bissau', 'гвинея-бисау', 'республика гвинея-бисау', '几内亚比绍', 'la república de guinea-bissau', 'the republic of guinea-bissau', 'غينيا - بيساو', 'la république de guinée-bissau'],
        "Guyana":['гайана', 'the co-operative republic of guyana', 'кооперативная республика гайана', 'guyana', 'غيانا', '圭亚那合作共和国', 'la república cooperativa de guyana', 'la république coopérative du guyana', '圭亚那', 'جمهورية غيانا التعاونية', 'republic of guyana'],
        "Haiti":['海地', '海地共和国', 'haití', 'haïti', 'haiti', 'la república de haití', 'جمهورية هايتي', 'هايتي', 'гаити', "la république d'haïti", 'республика гаити', 'the republic of haiti', 'republic of haiti'],
        "Jordan":['jordania','hashemite jordan', 'jordan', 'иорданское хашимитское королевство', 'иордания', 'jordanie', 'hashemite kingdom of jordan', 'the hashemite kingdom of jordan', '约旦哈希姆王国', 'el reino hachemita de jordania', 'المملكة الأردنية الهاشمية', 'hashemite jordan', 'le royaume hachémite de jordanie', 'الأردن', '约旦', 'jordan'],
        "Holy See":['vatican-city', '罗马教廷', 'le saint-siège', 'святой престол', 'santa sede', 'la santa sede', 'holy see', 'الكرسي الرسولي', 'the holy see', 'saint-siège'],
        "Honduras":['جمهورية هندوراس', '洪都拉斯', '洪都拉斯共和国', 'гондурас', 'هندوراس', 'the republic of honduras', 'la république du honduras', 'республика гондурас', 'honduras', 'republic of honduras', 'la república de honduras'],
        "Hong Kong":['hong kong s.a.r. of china','hong kong','hong kong (chine)', "Hong Kong Special Administrative Region of the People's Republic of China", '中華人民共和國香港特別行政區'],
        "Hungary":['la hongrie', 'hungaria', 'hungria', 'hungría', '匈牙利', 'hongrie', 'hungary', 'هنغاريا', 'венгрия'],
        "Iceland":['islandia', 'iceland', "la république d'islande", 'islande', 'جمهورية آيسلندا', '冰岛', 'آيسلندا', 'the republic of iceland', 'la república de islandia', 'исландия', '冰岛共和国', 'республика исландия', 'republic of iceland'],
        "Independent State of Papua New Guinea":['независимое государство папуа — новая гвинея', "l'état indépendant de papouasie-nouvelle-guinée", 'el estado independiente de papua nueva guinea', '巴布亚新几内亚', 'دولة بابوا غينيا الجديدة المستقلة', 'папуа — новая гвинея', 'independent state of papua new guinea', '巴布亚新几内亚独立国', 'papua new guinea', 'papua nueva guinea', 'papouasie-nouvelle-guinée', 'the independent state of papua new guinea', 'بابوا غينيا الجديدة'],
        "Independent State of Samoa":['el estado independiente de samoa', '萨摩亚', 'samoa', '萨摩亚独立国', 'دولة ساموا المستقلة', 'самоа', 'independent state of samoa', 'ساموا', 'независимое государство самоа', 'the independent state of samoa', "l'état indépendant du samoa"],        
        "India":['india', '印度', 'جمهورية الهند', 'inda', 'индия', 'la república de la india', '印度共和国', 'الهند', 'the republic of india', 'indian-subcontinent', "la république de l'inde", 'republic of india', 'inde', 'республика индия'],
        "Indonesia":['индонезия', 'جمهورية إندونيسيا', 'indonésie', 'indonesia', 'republic of indonesia', 'the republic of indonesia', 'إندونيسيا', 'республика индонезия', '印度尼西亚', 'la república de indonesia', "la république d'indonésie", '印度尼西亚共和国'],
        "Islamic Republic of Iran":['islamic republic of iran','Iran', 'iran',"la république islamique d'iran", 'irán', 'иран', 'la república islámica del irán', 'исламская республика иран', 'iran', 'جمهورية إيران الإسلامية', 'the islamic republic of iran', '伊朗伊斯兰共和国', 'إيران'],
        "Iraq":['iraqi-kurdistan','kurdistan irakien','la república del iraq', 'республика ирак', 'the republic of iraq', '伊拉克', "la république d'iraq", 'iraq', 'جمهورية العراق', 'republic of iraq', '伊拉克共和国', 'العراق', 'ирак'],
        "Ireland":['irland', 'ирландия', 'ireland', 'أيرلندا', '爱尔兰', 'irland-en-de', 'irlande', 'irlanda', "l'irlande"],
        "Israel":['ישראל', 'israël'],
        "Italy":['italia', 'italien', 'italy', 'andria', 'italie','italian', 'italie', 'italia', 'la république italienne', 'the republic of italy', 'la república italiana', '意大利共和国', 'جمهورية إيطاليا', 'итальянская республика', 'إيطاليا', '意大利', 'италия', 'italy', 'italian republic'],
        "Republic of Côte d'Ivoire":["ivory coast", "republic of côte d'ivoire","ivory coast", 'cote-d-ivoire',"côte d'ivoire","the republic of côte d'ivoire", 'республика кот-д’ивуар', 'كوت ديفوار', '科特迪瓦', '科特迪瓦共和国', "republic of côte d'ivoire", "côte d'ivoire", 'جمهورية كوت ديفوار', 'la república de côte d’ivoire', 'côte d’ivoire', "la république de côte d'ivoire", 'кот-д’ивуар'],
        "Jamaica":['جامايكا', 'jamaica', 'jamaïque', 'ямайка', 'la jamaïque', '牙买加'],
        "Japan":['日本国', 'اليابان', 'japón', 'le japon', 'japon', 'japan', 'япония', '日本', 'el japón'],
        "Jordan":['الأردن', 'لأردن', 'jordanie'],
        "Kazakhstan":['the republic of kazakhstan', '哈萨克斯坦共和国', 'كازاخستان', 'جمهورية كازاخستان', 'la república de kazajstán', 'республика казахстан', 'казахстан', 'kazakhstan', 'la république du kazakhstan', 'kazajstán', 'republic of kazakhstan', '哈萨克斯坦'],
        "Kenya":['كينيا', 'republic of kenya', '肯尼亚', 'кения', 'the republic of kenya', '肯尼亚共和国', 'республика кения', 'kenya', 'la república de kenya', 'جمهورية كينيا', 'la république du kenya'],
        "Kiribati":['кирибати', 'كيريباس', 'the republic of kiribati', 'la república de kiribati', 'la république de kiribati', 'республика кирибати', 'جمهورية كيريباس', 'kiribati', 'republic of kiribati', '基里巴斯共和国', '基里巴斯'],
        "Kosovo":['kosovo', 'Kosovo'],
        "Kuwait":['koweït'],
        "Kyrgyzstan":['kyrgyz', 'kirghizistan','kyrgyz republic', '吉尔吉斯共和国', 'kirguistán', '吉尔吉斯斯坦', 'the kyrgyz republic', 'кыргызская республика', 'kyrgyzstan', 'кыргызстан', 'جمهورية قيرغيزستان', 'قيرغيزستان', 'kirghizistan', 'la república kirguisa', 'la république kirghize', 'kyrgyz'],
        "Lao People's Democratic Republic":["lao people's democratic republic",'Laos','laos',"lao people's democratic",'la république démocratique populaire lao', 'جمهورية لاو الديمقراطية الشعبية', 'république démocratique populaire lao', "lao people's democratic", 'la república democrática popular lao', '老挝人民民主共和国', "lao people's democratic republic", "the lao people's democratic republic", 'лаосская народно-демократическая республика', 'república democrática popular lao'],
        "Latvia":['republic of latvia', 'جمهورية لاتفيا', 'letonia', 'the republic of latvia', 'la república de letonia', '拉脱维亚共和国', 'لاتفيا', 'latvia', '拉脱维亚', 'lettonie', 'la république de lettonie', 'латвия', 'латвийская республика'],
        "Lebanese":['lebanese republic', 'ливан', 'الجمهورية اللبنانية', '黎巴嫩', 'لبنان', 'la république libanaise', 'líbano', 'ливанская республика', '黎巴嫩共和国', 'lebanese', 'the lebanese republic', 'la república libanesa', 'lebanon', 'liban'],
        "Lebanon":['liban', "lebanese", 'lebanese republic', 'ливан', 'الجمهورية اللبنانية', '黎巴嫩', 'لبنان', 'la république libanaise', 'líbano', 'ливанская республика', '黎巴嫩共和国', 'lebanese', 'the lebanese republic', 'la república libanesa', 'lebanon', 'liban'],
        "Lesotho":['ليسوتو', 'kingdom of lesotho', '莱索托王国', 'مملكة ليسوتو', 'the kingdom of lesotho', 'lesotho', '莱索托', 'лесото', 'королевство лесото', 'le royaume du lesotho', 'el reino de lesotho'],
        "Liberia":['利比里亚共和国', 'the republic of liberia', 'libéria', '利比里亚', 'جمهورية ليبريا', 'republic of liberia', 'la république du libéria', 'республика либерия', 'либерия', 'la república de liberia', 'ليبريا', 'liberia'],
        "Libya":['利比亚', "l'état de libye", 'libye', 'государство ливия', 'دولة ليبيا', '利比亚国', 'libya', 'ливия', 'el estado de libia', 'libia', 'ليبيا', 'the state of libya'],
        "Lithuania":['lituania', 'ليتوانيا', 'la república de lituania', 'جمهورية ليتوانيا', 'lithuania', '立陶宛', 'la république de lituanie', ' 立陶宛共和国', 'republic of lithuania', 'the republic of lithuania', 'литовская республика', 'lituanie', 'литва'],
        "Luxembourg":['luxemburgo'],
        "Madagascar":['the republic of madagascar', 'madagascar', 'республика мадагаскар', 'republic of madagascar', 'la república de madagascar', '马达加斯加共和国', 'جمهورية مدغشقر', 'мадагаскар', 'la république de madagascar', 'مدغشقر', '马达加斯加'],
        "Malawi":['ملاوي', 'the republic of malawi', '马拉维', 'la república de malawi', 'республика малави', '马拉维共和国', 'malawi', 'جمهورية ملاوي', 'малави', 'republic of malawi', 'la république du malawi'],
        "Malaysia":['malay', 'malaysia', '马来西亚', 'ماليزيا', 'la malaisie', 'malaisie', 'малайзия', 'malasia'],
        "Maldives":['maldives', '马尔代夫', 'maldivas', 'the republic of maldives', 'جمهورية ملديف', 'republic of maldives', '马尔代夫共和国', 'ملديف', 'la république des maldives', 'la república de maldivas', 'мальдивские острова', 'мальдивская республика'],
        "Mali":['republic of mali', 'جمهورية مالي', 'malí', 'mali', 'la república de malí', 'республика мали', 'مالي', 'мали', '马里共和国', 'the republic of mali', 'la république du mali', '马里'],
        "Malta":['the republic of malta', 'мальта', 'مالطة', 'республика мальта', '马耳他共和国', 'جمهورية مالطة', 'la república de malta', 'republic of malta', 'malte', 'la république de malte', 'malta', '马耳他'],
        "Marshall Islands":['جزر مارشال', 'la république des îles marshall', 'îles marshall', '马绍尔群岛', 'the republic of the marshall islands', '马绍尔群岛共和国', 'جمهورية جزر مارشال', 'la república de las islas marshall', 'islas marshall', 'маршалловы острова', 'republic of the marshall islands', 'marshall islands', 'республика маршалловы острова'],
        "Martinique":['martinica'],
        "Mauritania":['исламская республика мавритания', 'الجمهورية الإسلامية الموريتانية', '毛里塔尼亚', 'mauritanie', 'the islamic republic of mauritania', 'la république islamique de mauritanie', 'islamic republic of mauritania', 'la república islámica de mauritania', 'мавритания', 'mauritania', 'موريتانيا', '毛里塔尼亚伊斯兰共和国'],
        "Mauritius":['موريشيوس', 'republic of mauritius', 'mauritius', 'جمهورية موريشيوس', '毛里求斯共和国', '毛里求斯', 'mauricio', 'the republic of mauritius', 'la république de maurice', 'maurice', 'la república de mauricio', 'маврикий', 'республика маврикий'],
        "Mexico":['mexique', 'mexixco', 'tijuana-baja-california'],
        "Micronesia":['федеративные штаты микронезии', 'los estados federados de micronesia', 'ولايات ميكرونيزيا الموحدة', 'les états fédérés de micronésie', 'the federated states of micronesia', 'micronésie', 'ميكرونيزيا', '密克罗尼西亚联邦', 'микронезия', 'micronesia'],
        "Republic of Moldova":['republic of moldova','moldova','république de moldova', 'moldavie', 'the republic of moldova', 'la république de moldova', 'la república de moldova', 'moldova', 'جمهورية مولدوفا', '摩尔多瓦共和国', 'república de moldova', 'республика молдова', 'republic of moldova'],
        "Mongolia":['монголия', 'منغوليا', 'mongolie', '蒙古', 'la mongolie', 'mongolia', '蒙古国'],
        "Montenegro":['الجبل الأسود', 'le monténégro', 'monténégro', '黑山', 'montenegro', 'черногория'],
        "Morocco":['marokko', '摩洛哥', 'el reino de marruecos', 'المملكة المغربية', 'the kingdom of morocco', 'марокко', 'maroc', 'королевство марокко', 'moroccov', '摩洛哥王国', 'kingdom of morocco', 'marruecos', 'le royaume du maroc', 'المغرب', 'morocco'],
        "Mozambique":['la république du mozambique', '莫桑比克共和国', 'جمهورية موزامبيق', 'موزامبيق', 'мозамбик', 'la república de mozambique', 'republic of mozambique', 'the republic of mozambique', 'республика мозамбик', '莫桑比克', 'mozambique'],
        "Myanmar":['缅甸联邦共和国', 'ميانمار', 'جمهورية اتحاد ميانمار', "la république de l'union du myanmar", 'республика союз мьянма', 'мьянма', 'myanmar', '缅甸', 'myanmar (birmanie)', 'the republic of the union of myanmar', 'republic of myanmar', 'la república de la unión de myanmar'],
        "Namibia":['namibia', 'جمهورية ناميبيا', 'республика намибия', 'republic of namibia', '纳米比亚共和国', 'намибия', 'the republic of namibia', 'ناميبيا', 'namibie', 'la república de namibia', '纳米比亚', 'la république de namibie'],
        "Nauru":['republic of nauru', '瑙鲁共和国', 'ناورو', '瑙鲁', 'nauru', 'республика науру', 'the republic of nauru', 'la république de nauru', 'науру', 'جمهورية ناورو', 'la república de nauru'],
        "Nepal":['le népal', '尼泊尔', 'nepal', 'népal', 'federal democratic republic of nepal', 'نيبال', 'непал'],
        "Netherlands":['niederlande', 'paises bajos', 'nederland', 'pays-bas', 'netherlands', "the netherlands", 'مملكة هولندا', 'le royaume des pays-bas', 'the netherlands', 'هولندا', '荷兰', 'the kingdom of the netherlands', 'netherlands', 'королевство нидерландов', 'kingdom of the netherlands', 'países bajos', 'pays-bas', '荷兰王国', 'el reino de los países bajos', 'нидерланды'],
        "New Caledonia":['nouvelle-caledonie'],
        "New Zealand":['новая зеландия', 'new-zealand-english', 'nouvelle-zélande', 'la nouvelle-zélande', 'new zealand', '新西兰', 'nueva zelandia', 'نيوزيلندا'],
        "Nicaragua":['la república de nicaragua', '尼加拉瓜共和国', 'نيكاراغوا', 'جمهورية نيكاراغوا', '尼加拉瓜', 'republic of nicaragua', 'nicaragua', 'никарагуа', 'the republic of nicaragua', 'la république du nicaragua', 'республика никарагуа'],
        "Niger":['尼日尔共和国', 'niger', 'the republic of the niger', 'níger', 'republic of the niger', 'нигер', '尼日尔', 'النيجر', 'республика нигер', 'la république du niger', 'la república del níger', 'جمهورية النيجر'],
        "Nigeria":['the federal republic of nigeria', 'la république fédérale du nigéria', 'федеративная республика нигерия', 'جمهورية نيجيريا الاتحادية', 'nigéria', 'نيجيريا', '尼日利亚联邦共和国', 'нигерия', '尼日利亚', 'la república federal de nigeria', 'nigeria', 'federal republic of nigeria'],
        "Niue":['niue', 'ниуэ', 'نيوي', '纽埃', 'nioué'],
        "North Macedonia":['macedonia','north macedonia', 'macédoine du nord', 'республика северная македония', 'مقدونيا الشمالية', 'la república de macedonia del norte', 'republic of north macedonia', '北马其顿共和国', 'северная македония', '北马其顿', 'the republic of north macedonia', 'republic-of-macedonia', 'macedonia del norte', 'جمهورية مقدونيا الشمالية', 'la république de macédoine du nord'],
        "Norway":['oslo', 'kingdom of norway', 'el reino de noruega', 'النرويج', 'королевство норвегия', 'مملكة النرويج', 'le royaume de norvège', 'the kingdom of norway', 'norway', '挪威', '挪威王国', 'noruega', 'норвегия', 'norvège'],
        "Pakistan":['пакистан', '巴基斯坦', '巴基斯坦伊斯兰共和国', 'исламская республика пакистан', 'pakistan', 'the islamic republic of pakistan', 'islamic republic of pakistan', 'pakistán', 'la république islamique du pakistan', 'la república islámica del pakistán', 'باكستان', 'جمهورية باكستان الإسلامية'],
        "Palau":['republic of palau', 'the republic of palau', 'палау', 'la república de palau', 'palau', 'республика палау', '帕劳', 'palaos', 'la république des palaos', 'بالاو', '帕劳共和国', 'جمهورية بالاو'],
        "Panama":['панама', 'panamá', 'بنما', 'республика панама', 'جمهورية بنما', 'panama', '巴拿马', 'republic of panama', 'la república de panamá', 'la république du panama', '巴拿马共和国', 'the republic of panama'],
        "Paraguay":['парагвай', '巴拉圭', 'republic of paraguay', 'جمهورية باراغواي', 'la república del paraguay', 'la république du paraguay', 'باراغواي', 'paraguay', 'республика парагвай', '巴拉圭共和国', 'the republic of paraguay'],
        "Bangladesh":["people's bangladesh","people's republic of bangladesh", "the people's republic of bangladesh", 'جمهورية بنغلاديش الشعبية', '孟加拉国', 'بنغلاديش', "people's bangladesh", 'народная республика бангладеш', 'la república popular de bangladesh', 'бангладеш', '孟加拉人民共和国', 'bangladesh', 'la république populaire du bangladesh'],
        "China":["people's china",'chine', 'الصين', 'la république populaire de chine', '中国', 'китайская народная республика', 'جمهورية الصين الشعبية', '中华人民共和国', 'la república popular china', "people's china", 'китай', 'china', "the people's republic of china", "people's republic of china"],
        "Peru":['秘鲁共和国', 'la république du pérou', 'republic of peru', 'la república del perú', 'перу', 'республика перу', 'pérou', 'perú', ' 秘鲁', 'peru', 'بيرو', 'جمهورية بيرو', 'the republic of peru'],
        "Philippines":['菲律宾共和国', 'la república de filipinas', 'филиппины', 'republic of the philippines', 'filipinas', 'جمهورية الفلبين', '菲律宾', 'الفلبين', 'the republic of the philippines', 'республика филиппины', 'philippines', 'la république des philippines'],
        "Poland":['poland-polski', 'poland-romania', 'republic of poland', 'la république de pologne', 'республика польша', 'pologne', 'جمهورية بولندا', 'the republic of poland', '波兰共和国', 'polen', 'polonia', 'польша', 'la república de polonia', '波兰', 'polska', 'بولندا', 'poland'],
        "Portugal":['portugal', 'portuguese', 'portuguese republic', '葡萄牙共和国', 'португалия', 'la república portuguesa', 'جمهورية البرتغال', '葡萄牙', 'la république portugaise', 'the portuguese republic', 'البرتغال', 'portuguese', 'portugal', 'португальская республика'],
        "Plurinational State of Bolivia":["l'état plurinational de bolivie", '多民族玻利维亚国', 'боливия', 'the plurinational state of bolivia', 'bolivia', 'bolivie', 'многонациональное государство боливия', 'بوليفيا', 'دولة بوليفيا المتعددة القوميات', 'el estado plurinacional de bolivia'],
        "Taiwan, Province of China":["taïwan", 'Taiwan',"taiwan, province of china", "taiwan", 'taiwan province of china'],
        "Principality of Andorra":['andorre', 'княжество андорра', '安道尔', 'andorra', 'principality of andorra', 'إمارة أندورا', 'андорра', 'أندورا', 'el principado de andorra', '安道尔公国', 'the principality of andorra', "la principauté d'andorre"],
        "Principality of Liechtenstein":['лихтенштейн', 'княжество лихтенштейн', '列支敦士登公国', 'liechtenstein', 'ليختنشتاين', 'el principado de liechtenstein', '列支敦士登', 'the principality of liechtenstein', 'la principauté du liechtenstein', 'principality of liechtenstein', 'إمارة ليختنشتاين'],
        "Principality of Monaco":['the principality of monaco', 'княжество монако', 'el principado de mónaco', 'principality of monaco', 'إمارة موناكو', 'monaco', 'موناكو', 'la principauté de monaco', '摩纳哥公国', '摩纳哥', 'mónaco', 'монако'],
        "Reunion":['la reunion', 'la-reunion'],
        "Romania":['румыния', '罗马尼亚', 'la roumanie', 'romanina', 'رومانيا', 'romaniaă', 'romania', 'roumanie', 'rumania'],
        "Russian Federation":["russian federation",'russia','russia-русский', 'fédération de russie', 'federación de rusia', 'الاتحاد الروسي', 'la federación de rusia', 'rusia', 'soviet-union', 'russie', '俄罗斯联邦', 'russian federation', 'the russian federation', 'la fédération de russie', 'российская федерация'],
        "Rwanda":['Rwandese','республика руанда', 'руанда', 'the republic of rwanda', 'la república de rwanda', '卢旺达', 'رواندا', 'la république du rwanda', 'rwanda', '卢旺达共和国', 'rwandese republic', 'rwandese', 'جمهورية رواندا'],
        "Saint Kitts and Nevis":['سانت كيتس ونيفس', 'saint kitts y nevis', 'saint-kitts-et-nevis', 'сент-китс и невис', 'saint kitts and nevis', ' 圣基茨和尼维斯'],
        "Saint Lucia":['سانت لوسيا', 'saint lucia', '圣卢西亚', 'sainte-lucie', 'сент-люсия', 'santa lucía'],
        "Saint Vincent and the Grenadines":['圣文森特和格林纳丁斯', 'сент-винсент и гренадины', 'san vicente y las granadinas', 'saint vincent and the grenadines', 'سانت فنسنت وجزر غرينادين', 'saint-vincent-et-les grenadines'],
        "San Marino":['saint-marin', 'сан-марино', 'جمهورية سان مارينو', 'سان مارينو', 'san marino', 'republic of san marino', 'la república de san marino', '圣马力诺共和国', 'the republic of san marino', 'республика сан-марино', '圣马力诺', 'la république de saint-marin'],
        "Saudi Arabia":['the kingdom of saudi arabia', 'саудовская аравия', '沙特阿拉伯王国', 'المملكة العربية السعودية', "le royaume d'arabie saoudite", 'королевство саудовская аравия', 'el reino de la arabia saudita', 'arabie saoudite', 'saudi arabia', '沙特阿拉伯', 'kingdom of saudi arabia', 'arabia saudita'],
        "Senegal":['塞内加尔', 'la república del senegal', '塞内加尔共和国', 'senegal', 'la république du sénégal', 'sénégal', 'сенегал', 'the republic of senegal', 'республика сенегал', 'السنغال', 'جمهورية السنغال', 'republic of senegal'],
        "Serbia":['serbie', 'la république de serbie', 'صربيا', 'республика сербия', 'جمهورية صربيا', 'la república de serbia', 'serbia', 'republic of serbia', 'сербия', '塞尔维亚共和国', 'the republic of serbia', '塞尔维亚'],
        "Seychelles":['republic of seychelles', 'республика сейшельские острова', 'the republic of seychelles', 'la république des seychelles', '塞舌尔', 'la república de seychelles', 'جمهورية سيشيل', '塞舌尔共和国', 'сейшельские острова', 'seychelles', 'سيشيل'],
        "Sierra Leone":['сьерра-леоне', 'جمهورية سيراليون', 'la república de sierra leona', 'sierra leona', 'سيراليون', 'sierra leone', 'the republic of sierra leone', 'la république de sierra leone', '塞拉利昂共和国', 'республика сьерра-леоне', 'republic of sierra leone', '塞拉利昂'],
        "Singapore":['سنغافورة', 'сингапур', 'the republic of singapore', 'республика сингапур', 'جمهورية سنغافورة', 'singapour', '新加坡共和国', 'la república de singapur', 'singapur', '新加坡', 'republic of singapore', 'singapore', 'la république de singapour'],
        "Slovakia":['slowakai', 'slovaquie', 'slovénie','slovak', 'eslovaquia', 'slovaquie', '斯洛伐克共和国', 'الجمهورية السلوفاكية', 'словацкая республика', 'slovak republic', 'the slovak republic', 'словакия', 'slovakia', '斯洛伐克', 'سلوفاكيا', 'la république slovaque', 'la república eslovaca'],
        "Slovenia":['la república de eslovenia', 'سلوفينيا', 'словения', 'the republic of slovenia', '斯洛文尼亚共和国', 'جمهورية سلوفينيا', 'slowenien', 'slovénie', '斯洛文尼亚', 'la république de slovénie', 'eslovenia', 'slovenia', 'республика словения', 'republic of slovenia'],     
        "Solomon Islands":['las islas salomón', '所罗门群岛', 'les îles salomon', 'solomon islands', 'соломоновы острова', 'îles salomon', 'islas salomón', 'جزر سليمان'],
        "Somalia":['somaliland region','somaliland','federal republic of somalia', 'الصومال', 'федеративная республика сомали', 'somalie', 'сомали', 'the federal republic of somalia', 'جمهورية الصومال الاتحادية', 'somalia', 'la república federal de somalia', '索马里联邦共和国', '索马里', 'la république fédérale de somalie', 'somaliland'],
        "South Africa":['the republic of south africa', 'la république sud-africaine', 'южно-африканская республика', 'جنوب أفريقيا', 'la república de sudáfrica', '南非共和国', 'جمهورية جنوب أفريقيا', 'sudáfrica', '南非', 'republic of south africa', 'south africa', 'южная африка', 'afrique du sud'],
        "Korea, Republic of":['South Korea','corée du sud', 'south korea' , "korea", 'la république populaire démocratique de corée', '大韩民国', 'korea-한국어', 'جمهورية كوريا الشعبية الديمقراطية', 'the republic of korea', 'república popular democrática de corea', 'جمهورية كوريا', 'la république de corée', '朝鲜民主主义人民共和国', "the democratic people's republic of korea", 'république populaire démocratique de corée', 'la república de corea', 'república de corea', 'republic of korea', 'республика корея', 'république de corée', 'корейская народно-демократическая республика', "democratic people's republic of korea", 'la república popular democrática de corea'],
        "Spain":['espa�a', 'espanha', 'kingdom of spain', 'españa', "le royaume d'espagne", '西班牙', 'spagna', 'the kingdom of spain', '西班牙王国', 'مملكة إسبانيا', 'королевство испания', 'испания', 'el reino de españa', 'espagne', 'إسبانيا', 'spanien', 'spain'],
        "State of Israel":['以色列', "l'état d'israël", '以色列国', 'دولة إسرائيل', 'إسرائيل', 'государство израиль', 'el estado de israel', 'israel', 'state of israel', 'the state of israel', 'израиль', 'israël'],
        "State of Kuwait":['الكويت', '科威特', 'kuwait', 'koweït', 'the state of kuwait', "l'état du koweït", 'государство кувейт', 'state of kuwait', '科威特国', 'دولة الكويت', 'el estado de kuwait', 'кувейт'],
        "the State of Palestine":['the state of palestine','west bank','Palestine', 'state of palestine','palestine','palestinian-territories', 'state of palestine', 'فلسطين', 'estado de palestina', 'دولة فلسطين', 'état de palestine', 'the state of palestine', "l'état de palestine", 'государство палестина', 'el estado de palestina', '巴勒斯坦国', 'palestinian territories'],
        "State of Qatar":['государство катар', 'قطر', '卡塔尔国', '卡塔尔', "l'état du qatar", 'state of qatar', 'катар', 'the state of qatar', 'el estado de qatar', 'qatar', 'دولة قطر'],
        "South Sudan":['republic of south sudan', 'جنوب السودان',"South Sudan", 'Soudan du Sud','soudan du sud', 'south sudan', 'the republic of south sudan', 'la república de sudán del sur', '南苏丹', 'южный судан', '南苏丹共和国', 'جمهورية جنوب السودان', 'республика южный судан', 'sudán del sur', 'république du soudan du sud', 'la république du soudan du sud'],
        "Sudan":['جمهورية السودان', '苏丹共和国', 'soudan', 'судан', 'the republic of the sudan', 'sudán', 'السودان', 'республика судан', '苏丹', 'la república del sudán', 'sudan', 'republic of the sudan', 'la république du soudan'],
        "Sultanate of Oman":['oman', 'omán', 'sultanate of oman', 'the sultanate of oman', 'عمان', '阿曼', 'султанат оман', 'la sultanía de omán', 'оман', "le sultanat d'oman", '阿曼苏丹国', 'سلطنة عمان'],
        "Suriname":['la república de suriname', '苏里南共和国', 'республика суринам', 'سورينام', 'the republic of suriname', 'suriname', 'جمهورية سورينام', 'la république du suriname', 'republic of suriname', 'суринам', '苏里南'],
        "Sweden":['瑞典', 'королевство швеция', 'el reino de suecia', '瑞典王国', 'schweden', 'kingdom of sweden', 'السويد', 'the kingdom of sweden', 'swaziland', 'suecia', 'sverige', 'suède', 'швеция', 'le royaume de suède', 'مملكة السويد', 'sweden'],
        "Swiss Confederation":['la confederación suiza', 'швейцарская конфедерация', 'swiss confederation', 'швейцария', 'الاتحاد السويسري', '瑞士 联邦', 'suiza', 'la confédération suisse', 'سويسرا', '瑞士', 'the swiss confederation', 'suisse', 'switzerland'],
        "Switzerland":['svizzera', 'suisse', 'szwajcaria', 'svizzera', 'schweiz', 'suiza', 'swiss'],
        "Syrian Arab Republic":['syrie', 'syria','Syria','syrian arab republic','الجمهورية العربية السورية', 'the syrian arab republic', 'la república árabe siria', 'república árabe siria', 'сирийская арабская республика', 'syrian arab', 'république arabe syrienne', '阿拉伯叙利亚共和国', 'syrian arab republic', 'la république arabe syrienne'],
        "Tajikistan":['la república de tayikistán', '塔吉克斯坦', 'جمهورية طاجيكستان', 'tadjikistan', 'la république du tadjikistan', 'tajikistan', 'the republic of tajikistan', 'республика таджикистан', '塔吉克斯坦共和国', 'таджикистан', 'طاجيكستان', 'republic of tajikistan', 'tayikistán'],
        "United Republic of Tanzania":['tanzanie',"united tanzania","tanzania", "United Republic of Tanzania", '坦桑尼亚联合共和国', 'la république-unie de tanzanie', 'la república unida de tanzanía', 'república unida de tanzanía', 'united republic of tanzania', 'united tanzania', 'جمهورية تنزانيا المتحدة', 'the united republic of tanzania', 'объединенная республика танзания', 'république-unie de tanzanie'],
        "Thailand":['تايلند', 'таиланд', '泰王国', 'thailande', 'the kingdom of thailand', 'thaïlande', 'kingdom of thailand', 'le royaume de thaïlande', 'thailand', 'королевство таиланд', 'مملكة تايلند', '泰国', 'tailandia', 'el reino de tailandia'],
        "Togo":['togolese','togo','la république togolaise', 'тоголезская республика', 'того', '多哥共和国', 'togolese republic', 'جمهورية توغو', 'togo', 'the togolese republic', 'la república togolesa', 'togolese', 'توغو', '多哥'],
        "Tonga":['汤加', 'el reino de tonga', '汤加王国', 'tonga', 'kingdom of tonga', 'тонга', 'the kingdom of tonga', 'le royaume des tonga', 'مملكة تونغا', 'تونغا', 'королевство тонга'],
        "Trinidad and Tobago":['trinidad-tobagot-english', 'trinité-et-tobago', 'trinidad & tobago',"trinidad and tobago",'republic of trinidad and tobago', 'جمهورية ترينيداد وتوباغو', 'la república de trinidad y tabago', 'trinidad y tabago', 'la république de trinité-et-tobago', '特立尼达和多巴哥共和国', 'trinidad and tobago', 'республика тринидад и тобаго', 'ترينيداد وتوباغو', 'тринидад и тобаго', 'the republic of trinidad and tobago', 'trinité-et-tobago', '特立尼达和多巴哥'],
        "Tunisia":['túnez', 'الجمهورية التونسية', 'la république tunisienne', 'تونس', 'тунис', '突尼斯', 'тунисская республика', 'tunisia', 'tunisie', '突尼斯共和国', 'republic of tunisia', 'la república de túnez', 'the republic of tunisia'],
        "Turkey":['турция', 'la république turque', 'la república de turquía', 'turquie', 'تركيا', 'турецкая республика', 'republic of turkey', 'جمهورية تركيا', '土耳其共和国', '土耳其', 'turkiye', 'turkey', 'the republic of turkey', 'turquía', 'birleşik-krallık-en-turkey'],
        "Turkmenistan":['turkménistan', 'turkmenistan', 'туркменистан', 'le turkménistan', '土库曼斯坦', 'turkmenistán', 'تركمانستان'],
        "Tuvalu":['图瓦卢', 'توفالو', 'тувалу', 'les tuvalu', 'tuvalu'],
        "United States Minor Outlying Islands":["us minor outlying islands",'u-s-minor-outlying-islands'],
        "Uganda":['أوغندا', 'ouganda', '乌干达共和国', 'جمهورية أوغندا', 'the republic of uganda', 'republic of uganda', 'уганда', 'la república de uganda', 'uganda', '乌干达', "la république de l'ouganda", 'республика уганда'],
        "Ukraine":['乌克兰', 'ucrania', 'ukraine', 'أوكرانيا', 'украина', "l'ukraine"],
        "Union of the Comoros":['comoros', 'comores', 'اتحاد جزر القمر', '科摩罗联盟', 'the union of the comoros', 'коморские острова', '科摩罗', 'جزر القمر', 'союз коморских островов', 'union of the comoros', "l'union des comores", 'comoras', 'la unión de las comoras'],
        "United Arab Emirates":['الإمارات العربية المتحدة', 'émirats arabes unis', 'united arab emirates', 'the united arab emirates', 'объединенные арабские эмираты', 'emiratos árabes unidos', '阿拉伯联合酋长国', 'los emiratos árabes unidos', 'les émirats arabes unis'],
        "United Kingdom":['royaume-uni', 'reino-unido', 'inglaterra', 'england', 'wales', "united great britain and northern ireland", "united great britain", "royaume-uni de grande-bretagne et d'irlande du nord", 'the united kingdom of great britain and northern ireland', 'el reino unido de gran bretaña e irlanda del norte', 'соединенное королевство великобритании и северной ирландии', 'united great britain and northern ireland', 'reino unido de gran bretaña e irlanda del norte', '大不列颠及北爱尔兰联合王国', 'united kingdom of great britain and northern ireland', "le royaume-uni de grande-bretagne et d'irlande du nord", 'المملكة المتحدة لبريطانيا العظمى وأيرلندا الشمالية'],
        "United Mexican States":['the united mexican states', 'الولايات المتحدة المكسيكية', 'mexico', 'المكسيك', 'united mexican states', 'les états-unis du mexique', 'méxico', 'мексика', 'mexique', '墨西哥', 'los estados unidos mexicanos', 'мексиканские соединенные штаты', '墨西哥合众国'],
        "United States":["united states of america", 'estados unidos', 'etats-unis', 'etats-unis', 'vereinigte-staaten-von-amerika', 'porto rico (états-unis)', 'états-unis', 'estados-unidos', 'vereinigte staaten von amerika', "les états-unis d'amérique", 'the united states of america', 'соединенные штаты америки', "états-unis d'amérique", '美利坚合众国', 'الولايات المتحدة الأمريكية', 'united states of america', 'los estados unidos de américa', 'estados unidos de américa'],
        "Uzbekistan":['узбекистан', '乌兹别克斯坦共和国', 'the republic of uzbekistan', "la république d'ouzbékistan", 'republic of uzbekistan', ' 乌兹别克斯坦', 'ouzbékistan', 'أوزبكستان', 'la república de uzbekistán', 'республика узбекистан', 'uzbekistán', 'جمهورية أوزبكستان', 'uzbekistan'],
        "Vanuatu":['瓦努阿图共和国', 'republic of vanuatu', 'فانواتو', 'the republic of vanuatu', 'вануату', 'республика вануату', 'la república de vanuatu', '瓦努阿图', 'جمهورية فانواتو', 'la république de vanuatu', 'vanuatu'],
        "Bolivarian Republic of Venezuela":['bolivarian republic of venezuela',"venezuela",'венесуэла', 'боливарианская республика венесуэла', 'the bolivarian republic of venezuela', 'Venezuela', 'la república bolivariana de venezuela', '委内瑞拉玻利瓦尔共和国', 'فنزويلا', 'la république bolivarienne du venezuela', 'venezuela', 'جمهورية فنزويلا البوليفارية'],        
        "Socialist Republic of Viet Nam":['viêt nam', 'vietnam', "socialist viet nam", '越南', 'the socialist republic of viet nam', '越南社会主义共和国', 'socialist viet nam', 'viet nam', 'فييت نام', 'socialist republic of viet nam', 'вьетнам', 'социалистическая республика вьетнам', 'جمهورية فييت نام الاشتراكية', 'la république socialiste du viet nam', 'la república socialista de viet nam'],
        "Yemen":['la république du yémen', 'الجمهورية اليمنية', 'اليمن', 'la república del yemen', 'yemen', '也门', 'yémen', 'йемен', 'йеменская республика', 'the republic of yemen', 'republic of yemen', '也门共和国'],
        "Yugoslavia":['yugoslavia', 'yugoslavie'],
        "Zambia":['赞比亚', 'جمهورية زامبيا', 'la república de zambia', 'republic of zambia', 'zambie', '赞比亚共和国', 'республика замбия', 'زامبيا', 'la république de zambie', 'zambia', 'the republic of zambia', 'замбия'],
        "Zimbabwe":['la república de zimbabwe', 'zimbabwe', 'جمهورية زمبابوي', 'the republic of zimbabwe', 'зимбабве', '津巴布韦共和国', 'республика зимбабве', 'la république du zimbabwe', '津巴布韦', 'زمبابوي', 'republic of zimbabwe'],
        "the State of Eritrea":['إريتريا', 'государство эритрея', 'the state of eritrea', 'érythrée', 'دولة إريتريا', '厄立特里亚国', 'el estado de eritrea', '厄立特里亚', 'eritrea', 'эритрея', "l'état d'érythrée"]
    }

_correspondance_with_official_name_lower_keys = defaultdict(list)


# ---------------------------------------------------------------------------------------------
# Correction des pays qui sont en erreur dans la librairie
lat_long = {('IT', 'EU'): (41.871940, 12.567380),  # Italie
            ('JP', 'AS'): (34.886306, 134.379711),  # ('JP', 'AS') Japan nan
            ('CZ', 'EU'): (49.817492, 15.472962),  # ('CZ', 'EU') Czech Republic nan
            ('VE', 'SA'): (6.423750, -66.589730),  # ('VE', 'SA') Venezuela nan
            ('NP', 'AS'): (28.394857, 84.124008),  # ('NP', 'AS') Nepal nan
            ('SY', 'AS'): (34.802075, 38.996815),  # ('SY', 'AS') Syria nan
            ('IE', 'EU'): (53.412910, -8.243890),  # ('IE', 'EU') Ireland nan
            ('UY', 'SA'): (-32.522779, -55.765835),  # ('UY', 'SA') Uruguay nan
            ('KY', 'NA'): (19.313300, -81.254600),  # ('KY', 'NA') Cayman Islands nan
            ('JO', 'AS'): (30.585164, 36.238414),  # ('JO', 'AS') Jordan nan
            ('ZW', 'AF'): (-19.015438, 29.154857),  # ('ZW', 'AF') Zimbabwe nan
            ('FI', 'EU'): (61.924110, 25.748151),  # ('FI', 'EU') Finland nan
            ('MW', 'AF'): (-13.254308, 34.301525),  # ('MW', 'AF') Malawi nan
            ('PY', 'SA'): (-23.442503, -58.443832),  # ('PY', 'SA') Paraguay nan
            ('UA', 'EU'): (44.874119, 33.151245),  # ('UA', 'EU') Ukraine nan
            ('EC', 'SA'): (-1.831239, -78.183406),  # ('EC', 'SA') Ecuador nan
            ('AM', 'AS'): (40.069099, 45.038189),  # ('AM', 'AS') Armenia nan
            ('LK', 'AS'): (7.873592, 80.773137),  # ('LK', 'AS') Sri Lanka nan
            ('PR', 'NA'): (18.220833, -66.590149),  # Puerto Rico
            ('GB', 'EU'): (52.3555177, -1.1743197),  # United Kingdom
            ('UG', 'AF'): (1.373333, 32.290275),  # ('UG', 'AF') Uganda nan
            ('GF', 'SA'): (3.921724136000023, -53.23312207499998),  # ('GF', 'SA') French Guiana nan
            ('PF', 'OC'): (-17.67739793399994, -149.40097329699998),  # French Polynesia nan
            ('GD', 'NA'): (12.151965053000026, -61.659644958999934),  # Grenada
            ('GY', 'SA'): (4.796422680000035, -58.97538657499996),  # Guyana
            ('JE', 'EU'): (49.21402591200007, -2.1327190749999545),  # Jersey
            ('MD', 'EU'): (47.20102827100004, 28.46370618900005),  # Moldova
            ('PW', 'OC'): (7.421479662000024, 134.511600068),  # Palau
            ('MF', 'NA'): (18.080477531000042, -63.06021562199999),  # Saint Martin
            ('SR', 'SA'): (3.9317774090000626, -56.01360780899995),  # Suriname
            ('VU', 'OC'): (-15.241355872999975, 166.8727570740001),  # Vanuatu
            ('AG', 'NA'): (17.0869391, -61.783491),
            ('BA', 'EU'): (43.9165389, 17.6721508),
            ('NL', 'NA'): (12.201890, -68.262383),
            ('CI', 'AF'): (7.5455112, -5.547545),  # Ivory Coast
            ('CW', 'NA'): (12.2135221, -68.9495816),
            ('CD', 'AF'): (-4.0335162, 21.7500603),
            ('FR', 'EU'): (46.71109, 1.7191036),
            ('IM', 'EU'): (54.2312716, -4.569504),
            ('MK', 'EU'): (41.6137143, 1.743258),  # North Macedonia
            ('RE', 'AF'): (-21.1306889, 55.5264794),
            ('TT', 'NA'): (10.4437128, -61.4191414),
            ('VI', 'NA'): (18.3434415, -64.8671634),
            ('BB', 'NA'): (13.1901325, -59.5355639),  # Barbados
            ('AW', 'NA'): (12.517572, -69.9649462),  # Aruba ('AW', 'NA') (nan, nan)
            ('GG', 'EU'): (49.4630653, -2.5881123),  # Guernesey
            ('PS', 'AS'): (31.947351, 35.227163),  # State of Palestine
            ("KN", "NA"): (17.2561791, -62.7019638),  # Saint Kitts and Nevis
            ("PM", "NA"): (46.9466881, -56.2622848),  # "Saint Pierre and Miquelon"
            ('VC', 'NA'): (13.252818, -61.197096),  # Saint Vincent And The Grenadines
            ('SX', 'NA'): (18.0347188, -63.0681114),  # Sint Maarten
            ('TT', 'SA'): (10.536421,-61.311951)      # 'Trinidad And Tobago'
            }

countries_dict = {}

# ---------------------------------------------------------------------------------------------
#                          Functions pour les données géographiques
# ---------------------------------------------------------------------------------------------
_manuelly_dic = {"kosovo": ("XK", "XXK","EU"),
                 "western sahara": ("EH", "ESH","AF"),
                 "reunion" : ("RE","REU", "AF"),
                 'central african':("CF","CAF", "AF"),
                 'state of palestine':("PS","PSE", "AS"),
                 'palestine':("PS","PSE", "AS"),
                 'democratic republic of the congo': ('CD', 'COD', 'AF'), 
                 'democratic sao tome and principe':("ST","STP", "AF"),
                 'sao tome and principe':("ST","STP", "AF"),
                 'tome and principe':("ST","STP", "AF"),
                 "timor-leste":("TL","TLS", "AS"),
                 "democratic timor-leste":("TL","TLS", "AS"),
                 'dominican':("DO","DOM", "NA"),
                 'syrian arab republic': ('SY','SYR', 'AS'),
                 "united states minor outlying islands": ('UM','UMI', 'NA'),
                 "yugoslavia": ('YU','YUG', 'EU'),
                 "holy see": ('VA','VAT', 'EU'),
                 "brazil": ('BR','BRA', 'SA'),
                 "plurinational state of bolivia":('BO', 'BOL', 'SA'),
                 'antigua and barbuda': ('AG', 'ATG' , 'NA'), 
                 'bosnia and herzegovina': ('BA', 'BIH', 'EU'), 
                 'caribbean netherlands': ('NL', 'BES', 'NA'), 
                 'cote d ivoire': ('CI','CIV', 'AF'), 
                 'curacao': ('CW','CUW', 'NA'), 
                 'Czech Republic' : ('CZ', 'CZE', 'EU'), # Czech Republic nan
                 'czech republic' : ('CZ', 'CZE','EU'), # Czech Republic nan
                 'isle of man': ('IM', 'IMN', 'EU'), 
                 'republic of macedonia': ('MK','MKD', 'EU'), 
                 'reunion': ('RE', 'REU','AF'), 
                 'saint kitts and nevis': ('KN', 'KNA', 'NA'), 
                 'saint pierre and miquelon': ('PM', 'SPM', 'NA'), 
                 'sint maarten': ('SX', 'SXM', 'NA'), 
                 'state of palestine': ('PS', 'PSE', 'AS'), 
                 'trinidad and tobago': ('TT', 'TTO', 'SA'), 
                 'united kingdom': ('GB', 'GBR', 'EU'), 
                 'virgin islands of the united states': ('VI', 'VIR', 'NA'), 
                 'maldives' : ('MV', 'MDV','AS'),
                 'united states of america':('US', 'USA','NA')
                 }

_country_to_correct_manually = list(_manuelly_dic.keys())

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                              FUNCTION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_dic_alpha2(country_name, verbose=0):
    if country_name is not None:
        return _manuelly_dic.get(country_name.lower(), None)
    return None

# 1. Conversion to Alpha 2 codes and Continents
def get_continent(country_name_param, include_format=False, verbose=False):
    """
    :param country_name_param (str): nom du pays recherché en anglais, attention, doit avoir des majuscules aux premières lettre de chaque mot, mais pas sur les petits mot
    :param include_format (boolean): True pour lancer le formatage (majuscules aux 1ère lettre et pas pour les petits mots du type : of,and, ...)
    :param verbose (boolean): True pour mode debug
    :return: (str, str) :(country a2 code, continent code)
    """
    country_name = country_name_param
    cn_a2_code = np.nan
    cn_a3_code = np.nan
    cn_continent = np.nan
    if country_name is not None and len(country_name) > 0:
        if include_format:
            country_name = country_name.title()
            country_name = country_name.replace(" Of ", " of ")
            country_name = country_name.replace(" The ", " the ")
            country_name = country_name.replace(" And ", " and ")
        try:
            cn_a2_code = country_name_to_country_alpha2(country_name)
        except:
            cn_a2_code = _manuelly_dic.get(country_name.lower(), np.nan)
            if cn_a2_code != np.nan:
                try:
                    cn_a2_code = cn_a2_code[0]
                except:
                    if verbose:
                        print("cn_a2_code ",country_name, "=> FAIL : ", cn_a2_code)
                    cn_a2_code = np.nan
        try:
            cn_a3_code = country_name_to_country_alpha3(country_name)
        except:
            cn_a3_code = _manuelly_dic.get(country_name.lower(), np.nan)
            if cn_a3_code != np.nan:
                try:
                    cn_a3_code = cn_a3_code[2]
                except:
                    if verbose:
                        print("cn_a3_code ",country_name, "=> FAIL alpha3 : ", cn_a3_code)
                    cn_a3_code = np.nan

        try:
            cn_continent = country_alpha2_to_continent_code(cn_a2_code)
        except:
            cn_continent = _manuelly_dic.get(country_name.lower(), np.nan)
            if cn_continent != np.nan:
                try:
                    cn_continent = cn_continent[1]
                except:
                    pass
    return cn_a2_code, cn_continent, cn_a3_code


def get_geolocation(country, geolocator=None, verbose=False):
    """
    :param country :(str, str)(country a2 code, continent code)
    :param geolocator: Nominatim
    :param verbose (boolean): True pour mode debug
    :return:(float, float):(latitude, longitude) or (nan, nan)
    """
    if geolocator is None:
        geolocator = Nominatim(user_agent="catuserbot")
    try:
        if country in lat_long.keys():
            return lat_long.get(country, np.nan)
        else:
            # Geolocate the center of the country
            loc = geolocator.geocode(country)
            # And return latitude and longitude
            return loc.latitude, loc.longitude
    except:
        # Return missing value
        return np.nan, np.nan
   

def get_country_alpha3(country_name, alpha2, verbose=False):
    alpha3 = None

    if alpha2 is not None:
        try:
            pcountry = pycountry.countries.get(alpha_2=alpha2)
            if pcountry is not None:
                alpha3 = pcountry.alpha_3
        except:
            pass

    if alpha3 is None:
        # On essaie avec le nom
        if country_name is not None:
            # On essaie avec le nom
            try:
                pcountry = pycountry.countries.get(name=country_name)
                if pcountry is not None:
                    alpha3 = pcountry.alpha_3
            except:
                pass
        
            if alpha3 is None:
                _, alpha3, _, _ = _manually_correct(country_name=country_name, alpha2=None, alpha3=None, official_name=None, continent_code=None, verbose=verbose)
            
        if alpha3 is None and verbose:
            print("cn_a3_code FAIL => ", country_name, alpha2)
    return alpha3

def get_country_official_name(country_name, alpha2=None, alpha3=None, verbose=False):
    official_name = None
    if alpha2 is not None:
        official_name = __get_country_official_name_with_alpha2(alpha2, verbose)
    
    if official_name is None and country_name is not None:
        official_name = __get_country_official_name_with_name(country_name, verbose)

    if official_name is None and alpha3 is not None:
        official_name = __get_country_official_name_with_alpha3(alpha3, verbose)

    return official_name
# ---------------------------------------------------------------------------------------------
#                               Préparation des données
# ---------------------------------------------------------------------------------------------
def get_country_data(country_name_param, geolocator=None, include_format=False, verbose=False, alpha3_param=None):
    """
    Récupère les données du pays
    :param country_name_param (str): nom du pays recherché en anglais, attention, doit avoir des majuscules aux premières lettre de chaque mot, mais pas sur les petits mot
    :param geolocator: Nominatim
    :param include_format (boolean): True pour lancer le formatage (majuscules aux 1ère lettre et pas pour les petits mots du type : of,and, ...)
    :param verbose (boolean): True pour mode debug
    :return: (str, str, float, float) : alpha2, continent_code, latitude, longitude
    """
    t0 = time()
    country_name = country_name_param
    alpha3 = alpha3_param
    country_id = np.nan
    alpha2 = None
    continent_code = None
    latitude = np.nan
    longitude = np.nan
    official_name = None

    try:
        if "Somaliland".lower() in country_name.lower():
            country_name = "Somalia"
    except:
        pass

    if alpha3_param is not None:
        try:
            pcountry = pycountry.countries.get(alpha_3=alpha3_param)
            if pcountry is not None:
                try:
                    alpha2 = pcountry.alpha_2
                    if alpha2 is not None:
                        continent_code = convert_country_alpha2_to_continent_code.country_alpha2_to_continent_code(alpha2)
                    else:
                        alpha2, alpha3, official_name, continent_code = _manually_correct(country_name, alpha2, alpha3_param, official_name, continent_code, verbose=verbose)
                except:
                    pass
                try:
                    official_name = pcountry.official_name
                except:
                    official_name = pcountry.name
                try:
                    country_id = pcountry.numeric
                except:
                    pass
        except:
            pass

    if alpha2 is None or continent_code is None:
        alpha2b, continent_codeb, alpha3b = get_continent(country_name, include_format, verbose)
        if alpha2 is None:
            alpha2 = alpha2b
        if continent_code is None:
            continent_code = continent_codeb
        if alpha3 is None:
            alpha3 = alpha3b

    if alpha3 is None and (alpha2 is not None or country_name is not None) :
        alpha3 = get_country_alpha3(country_name, alpha2)

    if official_name is None:
        official_name = get_country_official_name(country_name, alpha2, alpha3)

    if country_name.lower() in _country_to_correct_manually:
        alpha2, alpha3, official_name, continent_code = _manually_correct(country_name, alpha2, alpha3, official_name, continent_code, verbose=verbose)
    
    # Récupération des coordonnées
    if alpha2 != np.nan and continent_code != np.nan:
        if geolocator is None:
            geolocator = Nominatim(user_agent="catuserbot")

        geoloc = get_geolocation((alpha2, continent_code), geolocator)
        if geoloc != np.nan:
            try:
                latitude = geoloc[0]
                longitude = geoloc[1]
            except TypeError:
                print("TypeError for :", (alpha2, continent_code), country_name, geoloc)
        else:
            print("Country not found geoloc :", (alpha2, continent_code), country_name)
    else:
        if country_name == 'Holy See':
            latitude = 41.902916
            longitude = 12.453389
        else:
            print("Country not known :", country_name)

    t1 = time() - t0
    if verbose:
        print("get_country_data", country_name,
              " in {0:.3f} secondes............................................... END".format(t1))
    return alpha2, continent_code, latitude, longitude, alpha3, official_name, country_id

# ---------------------------------------------------------------------------------------------
#                               MAIN
# ---------------------------------------------------------------------------------------------

def _manually_correct(country_name, alpha2, alpha3, official_name, continent_code, verbose=0):
    try:
        if alpha2 is None or len(str(alpha2)) == 0 or np.isnan(alpha2):
            alpha2 = _manuelly_dic.get(country_name.lower(), (np.nan, np.nan, np.nan))[0]
    except Exception as error:
        if verbose:
            print(f"alpha2 = {alpha2}=>{error}")
    try:
        if alpha3 is None or len(str(alpha3)) == 0 or '-99' in str(alpha3) or np.isnan(alpha3):
            alpha3 = _manuelly_dic.get(country_name.lower(), (np.nan, np.nan, np.nan))[1]
    except Exception as error:
        if verbose:
            print(f"alpha3 = {alpha3}=>{error}")
    try:
        if official_name is None or len(str(official_name)) == 0 or np.isnan(official_name):
            official_name = country_name
    except Exception as error:
        if verbose:
            print(f"official_name = {official_name}=>{error}")
    try:
        if continent_code is None or len(str(continent_code)) == 0 or np.isnan(continent_code):
            continent_code =  _manuelly_dic.get(country_name.lower(), (np.nan, np.nan, np.nan))[2]
    except Exception as error:
        if verbose:
            print(f"continent_code = {continent_code}=>{error}")
    return alpha2, alpha3, official_name, continent_code


def __get_country_official_name_with_name(country_name, verbose=False, correct=False, lower=False):
    official_name = None
    if country_name is not None:
        # On essaie avec le nom
        try:
            pcountry = pycountry.countries.get(name=country_name)
            if pcountry is not None:
                try:
                    official_name = pcountry.official_name
                except:
                    official_name = pcountry.name
        except:
            pass

        # On recherche avec le nom corrigé
        if official_name is None and not correct:
            temp = _correct_official_name(country_name, verbose=verbose)
            if temp != country_name:
                official_name = __get_country_official_name_with_name(temp, verbose=verbose, correct=True)
                if official_name is None:
                    official_name = temp
        
        # On recherche avec le nom en minuscule
        if official_name is None and not lower:
            official_name = __get_country_official_name_with_name(country_name.lower(), verbose=verbose, correct=True, lower=True)
            
    if official_name is None:
        if correct and not lower:
            official_name = country_name
        elif verbose and not lower:
            print("official_name FAIL => ", country_name)
    return official_name


def _correct_official_name(country_name, verbose=0):   
    if len(_correspondance_with_official_name_lower_keys) == 0:
        for keys, value in countries_possibilities.items():
            for val in value:
                _correspondance_with_official_name_lower_keys[val] = keys
    
    if country_name is not None:
        return _correspondance_with_official_name_lower_keys.get(country_name.lower(),country_name)
    return None

def __get_country_official_name_with_alpha3(alpha3, verbose=False):
    official_name = None
    if alpha3 is not None:
        # On essaie avec le nom
        try:
            pcountry = pycountry.countries.get(alpha_3=alpha3)
            if pcountry is not None:
                try:
                    official_name = pcountry.official_name
                except:
                    official_name = pcountry.name
        except:
            pass
    if official_name is None and verbose:
        print("official_name with alpha3 FAIL => ", alpha3)
    return official_name


def __get_country_official_name_with_alpha2(alpha2, verbose=False):
    official_name = None
    if alpha2 is not None:
        try:
            pcountry = pycountry.countries.get(alpha_2=alpha2)
            if pcountry is not None:
                try:
                    official_name = pcountry.official_name
                except:
                    official_name = pcountry.name
        except:
            pass
    if official_name is None and verbose:
        print("official_name with alpha2 FAIL => ", alpha2)

    return official_name

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                              TESTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from os import getcwd
import pandas as pd

if __name__ == "__main__":

    verbose = 0
    
    # Récupère le répertoire du programme
    file_path = getcwd() + "\\"
    data_set_path = file_path + "dataset\\"
    print(f"Current execution path : {file_path}")
    print(f"Dataset path : {data_set_path}")

    data_set_filename = r"C:\Users\User\WORK\workspace-ia\PERSO\ara_commons\countries\country_engilsh_to_FSRCA-2.csv"
    
    df_origine = pd.read_csv(data_set_filename, sep=',')
    
    labels = list(df_origine.columns)
    labels.remove('English short')

    country_en_list = df_origine['English short'].values
    to_remove = {"Arab Republic of ", "Republic of the ", "Democratic Republic Of The ", "Democratic Republic Of ", "Democratic People's Republic of ", 'Federal Democratic Republic of ', 'Federal Republic of ', 'Islamic Republic of ', 'Kingdom of ', 'Republic of ', "Republic of "}

    for country_en in  tqdm(country_en_list):
        country_en_c = country_en.split("(")[0]
        country_en_c = country_en_c.split(" *")[0].strip()
        tlp1 = set()
        tlp1.add(country_en_c.lower())
        
        # Nettoyage du nom
        for st in to_remove:
            country_en_c = country_en_c.replace(st, "").strip()

        country_en_c = country_en_c.split(" Republic")[0].strip()
        tlp1.add(country_en_c.lower())
        
        off_name_start = get_country_official_name(country_en_c)
        if off_name_start is None:
            off_name_start = country_en_c

        if off_name_start is not None:

            tlp1.add(off_name_start.lower())
            off_name = off_name_start
            # Nettoyage du nom
            for st in to_remove:
                off_name = off_name.replace(st, "").strip()
            off_name = off_name.split(" Republic")[0].strip()
            tlp1.add(off_name.lower())

            tpl2 = set(countries_possibilities.get(off_name, []))
            tpl = tpl2.union(tlp1)

            for col in labels:
                name = df_origine.loc[df_origine['English short']==country_en, col].values[0]
                name = name.split(" (")[0]
                name = name.split(" *")[0].strip()
                tpl.add(name.lower())
            countries_possibilities[off_name] = list(tpl)
    
    keys = sorted(countries_possibilities.keys())
    for k in keys:
        print(f'"{k}":{countries_possibilities[k]},')




    test_list = ["Finland", "Finlande", "France", "Finland","Denmark","Norway","Iceland","Netherlands","Switzerland","Sweden","New Zealand","Canada","Austria","Australia","Costa Rica","Israel","Luxembourg","United Kingdom","Ireland",
"Germany","Belgium","United States","Czech Republic","United Arab Emirates","Malta","Mexico","France","Taiwan","Chile","Guatemala","Saudi Arabia","Qatar","Spain","Panama","Brazil",
"Uruguay","Singapore","El Salvador","Italy","Bahrain","Slovakia","Trinidad & Tobago","Poland","Uzbekistan","Lithuania","Colombia","Slovenia","Nicaragua","Kosovo","Argentina","Romania",
"Cyprus","Ecuador","Kuwait","Thailand","Latvia","South Korea","Estonia","Jamaica","Mauritius","Japan","Honduras","Kazakhstan","Bolivia","Hungary","Paraguay","Northern Cyprus",
"Peru","Portugal","Pakistan","Russia","Philippines","Serbia","Moldova","Libya","Montenegro","Tajikistan","Croatia","Hong Kong","Dominican RepublicvBosnia and Herzegovina",
"Turkey","Malaysia","Belarus","Greece","Mongolia","North Macedonia","Nigeria","Kyrgyzstan","Turkmenistan","Algeria","Moroccov","Azerbaijan","Lebanon","Indonesia","China",
"Vietnam","Bhutan","Cameroon","Bulgaria","Ghana","Ivory Coast","Nepal","Jordan","Benin","Congo (Brazzaville)","Gabon","Laos","South Africa","Albania","Venezuela","Cambodia",
"Palestinian Territories","Senegal","Somalia","Namibia","Niger","Burkina Faso","Armenia","Iran","Guinea","Georgia","Gambia","Kenya","Mauritania","Mozambique","Tunisia",
"Bangladesh","Iraq","Congo (Kinshasa)","Mali","Sierra Leone","Sri Lanka","Myanmar","Chad","Ukraine","Ethiopia","Swaziland","Uganda","Egypt","Zambia","Togo","India","Liberia",
"Comoros","Madagascar","Lesotho","Burundi","Zimbabwe","Haiti","Botswana","Syria","Malawi","Yemen","Rwanda","Tanzania","Afghanistan","Central African Republic","South Sudan", 'Bolivia', 'Democratic Republic Of The Congo',
'Eswatini, Kingdom of', 'Iran', 'Laos', 'Moldova', 'South Korea', 'State of Palestine', 'Syria', 'Taiwan', 'Tanzania', 'Venezuela','Vietnam']

    for country_feature in tqdm(test_list):
        off = __get_country_official_name_with_name(country_feature, verbose=verbose)
        if off is None:
            alpha2, continent_code, latitude, longitude, alpha3, off, country_id = get_country_data(country_feature, verbose=verbose)
            if off is None:
                print(f"\nASK : {country_feature} => GET : {off} => : {alpha2, continent_code, latitude, longitude, alpha3, off, country_id}")





