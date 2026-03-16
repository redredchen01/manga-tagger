"""Enhanced Tag Migration Script

為標籤庫生成完整的描述拓展，包括：
- visual_cues: 視覺特徵關鍵詞
- related_tags: 相關標籤
- negative_cues: 負向線索
- aliases: 別名
- enhanced_description: 增強描述

此腳本會批量處理所有標籤，提升RAG比對精確度。
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedTagMigrator:
    """增強版標籤遷移器"""

    def __init__(self):
        """初始化遷移器"""
        
        # ===== 1. 視覺線索詞典 =====
        self.visual_cue_dict = {
            # 角色特徵
            "耳朵": ["貓耳", "犬耳", "狐耳", "兔耳", "熊耳", "精靈耳", "長耳", "尖耳", "圓耳"],
            "尾巴": ["貓尾", "犬尾", "狐尾", "兔尾", "蓬鬆尾巴", "長尾巴", "尾巴"],
            "翅膀": ["天使翅膀", "惡魔翅膀", "蝙蝠翅膀", "龍翅膀", "昆蟲翅膀", "翅膀"],
            "角": ["惡魔角", "牛角", "鹿角", "犀牛角", "獨角", "角"],
            "眼睛": ["貓眼", "狐眼", "瞳孔", "異色瞳", "紅眼", "金眼", "大眼", "眼睛"],
            "頭髮": ["長髮", "短髮", "雙馬尾", "單馬尾", "捲髮", "直髮", "頭髮"],
            "胸部": ["巨乳", "貧乳", "大胸部", "小胸部", "胸部", "乳房"],
            "身體": ["尾巴", "翅膀", "角", "耳朵", "皮膚", "身體"],
            "牙齒": ["尖牙", "虎牙", "牙齒", "牙"],
            
            # 服裝配飾
            "制服": ["校服", "護士服", "女僕服", "警察制服", "軍裝", "制服"],
            "泳裝": ["比基尼", "連體泳衣", "泳裝", "泳衣"],
            "內衣": ["內衣", "內褲", "胸罩", "蕾絲內衣", "內衣"],
            "和服": ["和服", "浴衣", "振袖", "日本傳統服裝"],
            "絲襪": ["絲襪", "吊帶襪", "過膝襪", "褲襪", "襪子"],
            "項圈": ["項圈", "頸環", "寵物項圈", "頸飾"],
            "眼鏡": ["眼鏡", "墨鏡", "護目鏡", "眼鏡"],
            "貓耳": ["貓耳髮箍", "貓耳髮夾", "貓耳頭飾", "貓耳"],
            "兔耳": ["兔耳髮箍", "兔耳頭飾", "兔耳"],
        }

        # ===== 2. 相關標籤關聯詞典 =====
        self.related_tags_dict = {
            # 動物娘系列
            "貓娘": ["動物娘", "獸耳", "貓耳", "貓尾", "貓瞳", "貓爪", "獸娘", "furry", "nekomimi"],
            "犬娘": ["動物娘", "獸耳", "狗耳", "狗尾", "獸娘", "furry", "doggirl"],
            "狐娘": ["動物娘", "獸耳", "狐耳", "狐尾", "獸娘", "furry", "狐狸精", "kitsune"],
            "兔女郎": ["兔耳", "兔尾", "兔娘", "bunny girl", "兔女郎裝", "吊帶襪", "性感"],
            "貓女郎": ["貓耳", "貓尾", "貓娘", "catgirl", "nekomimi"],
            
            # 蘿莉/少女系列
            "蘿莉": ["小女孩", "未成年", "年輕女孩", "貧乳", "嬌小", "loli", "蘿莉塔"],
            "人妻": ["熟女", "御姐", "已婚", "成熟女性", "milf", "妻子"],
            "老太婆": ["熟女", "老女人", "old lady", "年長女性"],
            "少女": ["年輕", "女孩", "女性", "學生", "處女"],
            "御姐": ["成熟", "大姐姐", "年長女性", "熟女"],
            
            # 胸部相關
            "巨乳": ["大胸部", "豐滿", "乳溝", "big breasts", "胸部", "爆乳"],
            "貧乳": ["小胸部", "平板", "flat chest", "胸部", "蘿莉"],
            "超乳": ["極大胸部", "巨大胸部", "huge breasts", "gigantic breasts", "巨乳"],
            "極乳": ["不可能的胸部", "超巨大", "巨乳", "胸部"],
            "乳交": ["paizuri", "胸部", "乳溝", "乳頭"],
            "母乳": ["哺乳", "乳汁", "milk", "母乳餵養", "哺乳"],
            
            # 身體改造
            "變身": ["transformation", "轉換", "改變", "變形", "身體變化"],
            "身體改造": ["body modification", "改造", "人工", "義肢", "機械化"],
            "石化": ["petrification", "雕像", "石像", "僵硬", "固定"],
            "透明": ["透明人", "invisible", "隱形", "看不見", "透明"],
            "多臂": ["多手", "extra arms", "手臂", "multiple arms"],
            "多胸部": ["multiple breasts", "多乳", "乳房", "extra breasts"],
            
            # 翅膀/飛行
            "翅膀": ["wing", "羽翼", "飛行", "天使", "惡魔", "鳥類"],
            "天使": ["翅膀", "光環", "halo", "神聖", "白色翅膀", "羽翼"],
            "惡魔": ["翅膀", "角", "尾巴", "demon", "地獄", "黑暗"],
            "鳥娘": ["翅膀", "鳥類", "羽翼", "harpy", "鳥人", "禽娘"],
            "蝙蝠女": ["翅膀", "蝠翼", "bat wings", "蝙蝠", "夜行性"],
            
            # 觸手/異形
            "觸手": ["tentacle", "觸手怪", "章魚", "異形", "纏繞"],
            "史萊姆": ["slime", "黏液", "凝膠", "果凍", "半透明"],
            "章魚": ["octopus", "觸手", "八爪魚", "頭足類"],
            "蜘蛛娘": ["蜘蛛", "spider", "蛛網", "八足", "arachnid"],
            
            # 肌肉/體格
            "肌肉": ["muscle", "健美", "發達", "二頭肌", "腹肌", "體格"],
            "大肌肉": ["巨大肌肉", "extreme muscle", "誇張肌肉", "muscle", "bodybuilding"],
            "肌肥大": ["肌肉生長", "muscle growth", "肌肉變大", "訓練"],
            
            # 屍體/不死
            "屍體": ["corpse", "死屍", "死人", "遺體", "屍姦"],
            "殭屍": ["zombie", "喪屍", "活死人", "不死", "walking dead"],
            "幽靈": ["ghost", "幽靈", "鬼", "spirit", "透明"],
            "吸血鬼": ["vampire", "吸血", "尖牙", "夜行者", "血族"],
            
            # 獸交/動物
            "獸交": ["bestiality", "人獸交", "動物交", "獸醫"],
            "動物娘": ["furry", "獸人", "獸耳", "獸尾", "animal girl"],
            "福瑞": ["furry", "獸人", "獸娘", "獸耳", "毛茸茸"],
            
            # 生殖/性行為
            "口交": ["oral", "口交", "blowjob", "吞精", "口腔"],
            "肛交": ["anal", "肛交", "後庭", "臀部"],
            "陰道交": ["vaginal", "陰道交", "正常性交", "插入"],
            "群交": ["gangbang", "群交", "多人", "輪姦"],
            
            # 服裝
            "校服": ["school uniform", "制服", "學生", "水手服", "學蘭"],
            "女僕": ["maid", "女僕裝", "圍裙", "管家", "仆娘"],
            "護士": ["nurse", "護士服", "白衣", "醫療", "護理"],
            "泳裝": ["swimsuit", "比基尼", "泳衣", "海邊"],
            "和服": ["kimono", "和服", "浴衣", "日本", "傳統"],
            "兔女郎": ["bunny girl", "兔女郎裝", "性感", "吊帶襪", "貓耳"],
            
            # 姿勢
            "後背位": ["doggy style", "後入式", "肛交", "狗爬式"],
            "正常位": ["missionary", "傳教士", "面對面", "標準"],
            "騎乘位": ["cowgirl", "上位", "騎乘", "跨坐"],
            "站立位": ["standing", "站立", "懸掛", "抱著"],
            
            # 主題
            "調教": ["training", "調教", "奴隸", "主人", "SM"],
            "凌辱": ["rape", "強姦", "非自願", "暴力", "侵犯"],
            "NTR": ["netorare", "寢取", "被綠", "他人蔔"],
        }

        # ===== 3. 負向線索詞典 =====
        self.negative_cues_dict = {
            "蘿莉": ["成熟女性", "巨乳", "豐滿身材", "成年女性", "年輕女孩（非蘿莉）", "普通兒童"],
            "人妻": ["未婚", "年輕女孩", "未成年", "小女孩"],
            "巨乳": ["貧乳", "平板", "小胸部", "蘿莉"],
            "貓娘": ["純人類", "無動物特徵", "無貓耳", "無貓尾", "普通貓"],
            "犬娘": ["純人類", "無動物特徵", "無狗耳", "無狗尾"],
            "狐娘": ["純人類", "無動物特徵", "無狐耳", "無狐尾"],
            "獸交": ["人類", "無動物參與", "furry交"],
            "屍姦": ["活人", "zombie", "活屍"],
            "殭屍": ["活人", "普通幽靈", "活屍"],
            "女僕": ["便服", "普通服裝", "非女僕裝"],
            "校服": ["便服", "泳裝", "內衣", "睡衣"],
        }

        # ===== 4. 英文別名詞典 =====
        self.aliases_dict = {
            "蘿莉": ["loli", "little girl", "young girl", "oppai loli", "low lolicon", "lolicon"],
            "人妻": ["milf", "mature woman", "wife", "mother", "married woman"],
            "老太婆": ["old lady", "elderly woman", "grandma", "old woman"],
            "少女": ["girl", "young woman", "maiden", "teenager"],
            "御姐": ["older sister type", "mature girl", "senpai", "big sister"],
            "巨乳": ["big breasts", "large breasts", "busty", "oppai", "bigger breasts"],
            "貧乳": ["small breasts", "flat chest", "tiny breasts", "flat"],
            "超乳": ["huge breasts", "gigantic breasts", "massive breasts", "enormous breasts"],
            "極乳": ["impossible breasts", "ridiculous breasts", "absurdly large"],
            "貓娘": ["catgirl", "nekomimi", "cat ear girl", "貓耳娘", "feline girl"],
            "犬娘": ["doggirl", "dog girl", "dog ear girl", "犬耳娘"],
            "狐娘": ["foxgirl", "fox girl", "kitsune", "狐耳娘"],
            "兔女郎": ["bunny girl", "rabbit girl", "playboy bunny", "兔耳娘"],
            "殭屍": ["zombie", "undead", "living dead", "walking dead"],
            "幽靈": ["ghost", "specter", "spirit", "apparition", "phantom"],
            "吸血鬼": ["vampire", "bloodsucker", "nosferatu", "dracula"],
            "惡魔": ["demon", "devil", "fiend", "daemon", "succubus"],
            "天使": ["angel", "cherub", "seraph", "heavenly being"],
            "女僕": ["maid", "servant", "housekeeper", "butler"],
            "護士": ["nurse", "doctor", "medical", "scrubs", "white coat"],
            "校服": ["school uniform", "uniform", "student uniform", "制服"],
            "泳裝": ["swimsuit", "bikini", "swimwear", "bathing suit", "泳衣"],
            "觸手": ["tentacle", "tentacles", "octopus", "kraken"],
            "史萊姆": ["slime", "gelatinous", "ooze", "jelly"],
            "乳交": ["paizuri", "titty fuck", "breast sex", "tits job"],
            "母乳": ["breast milk", "nursing", "lactation", "milking"],
            "肌肉": ["muscle", "muscular", "buff", "fit", "athletic"],
            "變身": ["transformation", "transform", "morph", "change"],
            "身體改造": ["body modification", "bodmod", "改造", "augmentation"],
            "調教": ["training", "discipline", "BDSM", "submission", "domination"],
            "凌辱": ["rape", "sexual assault", "forced", "non-consensual"],
            "NTR": ["netorare", "cuckold", "cheating", "infidelity"],
            "獸交": ["bestiality", "zoophilia", "animal sex"],
            "動物娘": ["furry", "furry girl", "animal girl", "anthro"],
            "福瑞": ["furry", "fursona", "anthropomorphic", "furry fan"],
            "後背位": ["doggy style", "rear entry", "back door", "from behind"],
            "正常位": ["missionary", "face to face", "traditional"],
            "騎乘位": ["cowgirl", "girl on top", "riding", "straddling"],
            "蜘蛛娘": ["spider girl", "arachnid girl", "spider woman"],
            "蛇娘": ["snake girl", "serpent girl", "naga", "reptile girl"],
            "蜥蜴娘": ["lizard girl", "reptile girl", "saurian girl"],
            "美人魚": ["mermaid", "merman", "aquatic", "魚人"],
            "半人馬": ["centaur", "centaure", "horse-human hybrid"],
            "機娘": ["mecha girl", "robot girl", "android", "cyborg"],
            "機器人": ["robot", "android", "cyborg", "machine", "mecha"],
            "精靈": ["elf", "elven", "fairy", "fae", "pixie"],
            "哥布林": ["goblin", "gob", "hobgoblin", "gremlin"],
            "史萊姆娘": ["slime girl", "gelatinous girl", "slime woman"],
        }

        # ===== 5. 分類關鍵詞 =====
        self.category_keywords = {
            "character": [
                "蘿莉", "少女", "人妻", "熟女", "御姐", "正太", "少年", "小男孩",
                "貓娘", "犬娘", "狐娘", "兔女郎", "獸人", "精靈", "魔物", "機娘",
                "天使", "惡魔", "幽靈", "殭屍", "喪屍", "機械人", "半人馬", "美人魚",
                "蛇娘", "蜥蜴娘", "蜘蛛娘", "烏賊娘", "章魚娘", "青蛙娘", "蝴蝶娘",
                "老虎", "獅子", "豹", "熊", "熊貓", "兔子", "老鼠", "松鼠",
                "哥布林", "矮人", "吸血鬼", "狼人", "生化人", "外星人", "宇航員",
                "教師", "醫生", "護士", "警察", "服務生", "女僕", "巫女", "修女",
            ],
            "clothing": [
                "校服", "泳裝", "内衣", "和服", "女僕", "護士", "警察", "體操服",
                "兔女郎", "旗袍", "巫女", "婚纱", "西裝", "禮服", "睡衣", "內褲",
                "胸罩", "絲襪", "吊帶襪", "過膝襪", "褲襪", "襯衫", "裙子", "褲子",
                "連衣裙", "大衣", "外套", "夾克", "毛衣", "T恤", "背心", "泳衣",
                "比基尼", "情趣內衣", "SM裝", "皮革", "乳膠", "緊身衣", "制服",
            ],
            "body": [
                "巨乳", "貧乳", "長腿", "絲襪", "吊帶襪", "過膝襪", "裸足", "眼鏡",
                "義眼", "義肢", "疤痕", "紋身", "翅膀", "角", "尾巴", "耳朵",
                "肌肉", "大肌肉", "胸部", "乳頭", "乳暈", "臀部", "嘴巴", "陰部",
                "陰莖", "陰道", "肛門", "乳交", "口交", "腋毛", "陰毛", "腹毛",
                "手指", "腳趾", "膝蓋", "肘部", "脖子", "肩膀", "背部", "腰部",
            ],
            "action": [
                "做愛", "口交", "手淫", "乳交", "足交", "肛交", "群交", "多人",
                "綁架", "監禁", "調教", "凌辱", "強姦", "痴漢", "騷擾", "侵犯",
                "自慰", "高潮", "射精", "排卵", "懷孕", "分娩", "哺乳", "吮吸",
                "觸摸", "親吻", "舔舐", "咬", "抓", "掐", "打", "踢",
                "站立", "坐下", "躺下", "跪下", "趴下", "彎腰", "轉身", "跳躍",
            ],
            "theme": [
                "純愛", "NTR", "凌辱", "調教", "奴隸", "主從", "百合", "耽美",
                "亂倫", "師生", "職場", "近親", "戀物", "戀足", "戀乳", "戀腋",
                "科幻", "奇幻", "魔法", "武術", "戰鬥", "冒險", "恐怖", "血腥",
                "幽靈", "詛咒", "魔法", "變身", "時間停止", "空間扭曲",
            ],
        }

    def categorize_tag(self, tag_name: str, description: str) -> str:
        """自動分類標籤"""
        text = f"{tag_name} {description}".lower()
        
        for category, keywords in self.category_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    return category
        
        return "other"

    def extract_visual_cues(self, tag_name: str, description: str) -> List[str]:
        """從標籤名稱和描述中提取視覺線索"""
        visual_cues = []
        text = f"{tag_name} {description}"
        
        # 從標籤名稱提取
        for cue_type, cues in self.visual_cue_dict.items():
            for cue in cues:
                if cue in tag_name or cue in description:
                    if cue not in visual_cues:
                        visual_cues.append(cue)
        
        return list(set(visual_cues))

    def infer_related_tags(self, tag_name: str, description: str) -> List[str]:
        """推斷相關標籤"""
        related_tags = []
        
        # 從預定義關聯詞典獲取
        if tag_name in self.related_tags_dict:
            related_tags.extend(self.related_tags_dict[tag_name])
        
        # 從描述中提取相關術語
        related_terms = [
            "乳交", "口交", "肛交", "陰道交", "自慰", "高潮", "射精",
            "胸部", "乳房", "乳頭", "乳暈", "陰莖", "陰道", "肛門",
            "蘿莉", "少女", "人妻", "熟女", "御姐", "女王", "奴隸",
            "調教", "凌辱", "強姦", "非自願", "自願", "戀愛",
            "貓耳", "犬耳", "狐耳", "兔耳", "尾巴", "翅膀", "角",
            "肌肉", "巨乳", "貧乳", "豐滿", "嬌小", "高挑",
            "制服", "內衣", "泳裝", "絲襪", "眼鏡", "項圈",
            "觸手", "史萊姆", "透明", "石化", "變身", "改造",
            "殭屍", "幽靈", "吸血鬼", "惡魔", "天使", "精靈",
        ]
        
        for term in related_terms:
            if term in description and term not in related_tags:
                related_tags.append(term)
        
        return list(set(related_tags))

    def generate_negative_cues(self, tag_name: str, description: str) -> List[str]:
        """生成負向線索"""
        negative_cues = []
        
        # 從預定義負向線索獲取
        if tag_name in self.negative_cues_dict:
            negative_cues.extend(self.negative_cues_dict[tag_name])
        
        # 從描述中的否定詞提取
        negation_words = ["不要", "不得", "不應", "不該", "不要", "別", "莫", "勿"]
        
        for word in negation_words:
            if word in description and f"'{word}'" not in str(negative_cues):
                negative_cues.append(f"包含'{word}'的場景")
        
        return list(set(negative_cues))

    def generate_aliases(self, tag_name: str, description: str) -> List[str]:
        """生成別名列表"""
        aliases = []
        
        # 從預定義別名獲取
        if tag_name in self.aliases_dict:
            aliases.extend(self.aliases_dict[tag_name])
        
        # 從描述中提取英文詞彙
        english_pattern = r'\b[a-zA-Z]{3,}\b'
        matches = re.findall(english_pattern, description)
        for match in matches:
            alias = match.strip()
            if alias and len(alias) > 2 and alias.lower() != tag_name.lower():
                if alias not in aliases:
                    aliases.append(alias)
        
        # 從標籤名稱生成拼音別名
        pinyin_map = {
            "蘿莉": ["loli", "luoli"],
            "人妻": ["renqi", "wife", "milf"],
            "巨乳": ["juru", "big breasts", "oppai"],
            "貧乳": ["pinru", "small breasts", "flat chest"],
            "乳交": ["rujiao", "paizuri"],
            "母乳": ["muru", "breast milk", "lactation"],
            "貓娘": ["maoniang", "catgirl", "nekomimi"],
            "犬娘": ["quanniang", "doggirl"],
            "狐娘": ["huniang", "foxgirl", "kitsune"],
            "兔女郎": ["tunvlang", "bunny girl"],
            "女僕": ["nvpu", "maid"],
            "護士": ["hushi", "nurse"],
            "校服": ["xiaofu", "school uniform", "uniform"],
            "泳裝": ["zhuangyong", "swimsuit", "bikini"],
            "觸手": ["chushou", "tentacle"],
            "史萊姆": ["shilaimu", "slime"],
            "殭屍": ["jiangshi", "zombie", "undead"],
            "幽靈": ["youling", "ghost", "spirit"],
            "吸血鬼": ["xixuegui", "vampire"],
            "惡魔": ["emo", "demon", "devil"],
            "天使": ["tianshi", "angel"],
            "精靈": ["jingling", "elf", "fairy"],
            "調教": ["tiaojiao", "training", "BDSM"],
            "凌辱": ["lingru", "rape"],
            "NTR": ["netorare", "cuckold"],
            "獸交": ["shoujiao", "bestiality"],
            "動物娘": ["dongwuniang", "furry"],
            "福瑞": ["furry", "fursona"],
        }
        
        if tag_name in pinyin_map:
            for alias in pinyin_map[tag_name]:
                if alias not in aliases:
                    aliases.append(alias)
        
        return list(set(aliases))

    def calculate_confidence_boost(self, tag_name: str, description: str) -> float:
        """計算信心度加成"""
        high_confidence = [
            "蘿莉", "人妻", "巨乳", "貧乳", "貓娘", "犬娘", "狐娘",
            "兔女郎", "女僕", "護士", "校服", "泳裝", "絲襪", "眼鏡",
            "殭屍", "幽靈", "吸血鬼", "惡魔", "天使", "精靈",
            "觸手", "史萊姆", "肌肉", "乳交", "母乳", "調教",
            "後背位", "正常位", "騎乘位", "站立位",
        ]
        
        medium_confidence = [
            "老太婆", "正太", "少年", "御姐", "熟女",
            "半人馬", "美人魚", "機娘", "機器人", "哥布林",
            "蜘蛛娘", "蛇娘", "蜥蜴娘", "青蛙娘", "章魚",
            "獸交", "動物娘", "福瑞", "NTR", "凌辱",
            "群交", "口交", "肛交", "陰道交",
        ]
        
        if tag_name in high_confidence:
            return 1.1
        elif tag_name in medium_confidence:
            return 1.05
        else:
            desc_length = len(description)
            if desc_length > 100:
                return 1.05
            elif desc_length > 50:
                return 1.02
            else:
                return 1.0

    def enhance_description(self, tag_name: str, description: str, 
                          visual_cues: List[str], 
                          related_tags: List[str]) -> str:
        """生成增強描述"""
        enhanced = description
        
        if visual_cues:
            cue_text = "視覺特徵：" + "、".join(visual_cues[:10]) + "。"
            if cue_text not in enhanced:
                enhanced = f"{enhanced}\n{cue_text}"
        
        if related_tags:
            related_text = "相關標籤：" + "、".join(related_tags[:10]) + "。"
            if related_text not in enhanced:
                enhanced = f"{enhanced}\n{related_text}"
        
        return enhanced.strip()

    def migrate_tags(self, input_path: str, output_path: str) -> List[Dict[str, Any]]:
        """執行標籤遷移"""
        logger.info(f"載入標籤庫: {input_path}")
        
        with open(input_path, "r", encoding="utf-8") as f:
            original_tags = json.load(f)
        
        logger.info(f"載入 {len(original_tags)} 個標籤")
        
        enhanced_tags = []
        for i, tag_data in enumerate(original_tags):
            tag_name = tag_data.get("tag_name", "")
            description = tag_data.get("description", "")
            
            if not tag_name:
                logger.warning(f"跳過無名標籤: {tag_data}")
                continue
            
            category = self.categorize_tag(tag_name, description)
            visual_cues = self.extract_visual_cues(tag_name, description)
            related_tags = self.infer_related_tags(tag_name, description)
            negative_cues = self.generate_negative_cues(tag_name, description)
            aliases = self.generate_aliases(tag_name, description)
            confidence_boost = self.calculate_confidence_boost(tag_name, description)
            enhanced_desc = self.enhance_description(tag_name, description, visual_cues, related_tags)
            
            enhanced_tag = {
                "tag": tag_name,
                "category": category,
                "description": enhanced_desc,
                "original_description": description,
                "visual_cues": visual_cues,
                "related_tags": related_tags,
                "negative_cues": negative_cues,
                "aliases": aliases,
                "confidence_boost": confidence_boost,
            }
            
            enhanced_tags.append(enhanced_tag)
            
            if (i + 1) % 100 == 0:
                logger.info(f"已處理 {i + 1}/{len(original_tags)} 個標籤")
        
        logger.info(f"保存 {len(enhanced_tags)} 個增強標籤到: {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(enhanced_tags, f, ensure_ascii=False, indent=2)
        
        logger.info("遷移完成!")
        return enhanced_tags

    def generate_report(self, enhanced_tags: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成遷移報告"""
        report = {
            "total_tags": len(enhanced_tags),
            "categories": {},
            "tags_with_visual_cues": 0,
            "tags_with_related_tags": 0,
            "tags_with_negative_cues": 0,
            "tags_with_aliases": 0,
            "average_visual_cues": 0,
            "average_related_tags": 0,
            "average_aliases": 0,
        }
        
        total_visual_cues = 0
        total_related_tags = 0
        total_aliases = 0
        
        for tag in enhanced_tags:
            category = tag.get("category", "other")
            report["categories"][category] = report["categories"].get(category, 0) + 1
            
            if tag.get("visual_cues"):
                report["tags_with_visual_cues"] += 1
                total_visual_cues += len(tag["visual_cues"])
            
            if tag.get("related_tags"):
                report["tags_with_related_tags"] += 1
                total_related_tags += len(tag["related_tags"])
            
            if tag.get("negative_cues"):
                report["tags_with_negative_cues"] += 1
            
            if tag.get("aliases"):
                report["tags_with_aliases"] += 1
                total_aliases += len(tag["aliases"])
        
        if report["tags_with_visual_cues"] > 0:
            report["average_visual_cues"] = round(total_visual_cues / report["tags_with_visual_cues"], 2)
        if report["tags_with_related_tags"] > 0:
            report["average_related_tags"] = round(total_related_tags / report["tags_with_related_tags"], 2)
        if report["tags_with_aliases"] > 0:
            report["average_aliases"] = round(total_aliases / report["tags_with_aliases"], 2)
        
        return report


def main():
    """主入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="增強版標籤庫遷移工具")
    parser.add_argument(
        "--input",
        default="./data/tags.json",
        help="原始標籤庫路徑 (默認: ./data/tags.json)"
    )
    parser.add_argument(
        "--output",
        default="./data/tags_enhanced.json",
        help="輸出增強標籤庫路徑 (默認: ./data/tags_enhanced.json)"
    )
    parser.add_argument(
        "--report",
        default="./data/enhancement_report.json",
        help="報告輸出路徑 (默認: ./data/enhancement_report.json)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="測試運行，不保存輸出"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"輸入文件不存在: {args.input}")
        return
    
    migrator = EnhancedTagMigrator()
    
    if args.dry_run:
        logger.info("測試運行模式（不保存輸出）")
        with open(input_path, "r", encoding="utf-8") as f:
            original_tags = json.load(f)
        logger.info(f"將遷移 {len(original_tags)} 個標籤")
        
        for tag_data in original_tags[:5]:
            tag_name = tag_data.get("tag_name", "Unknown")
            description = tag_data.get("description", "")
            
            category = migrator.categorize_tag(tag_name, description)
            visual_cues = migrator.extract_visual_cues(tag_name, description)
            related_tags = migrator.infer_related_tags(tag_name, description)
            aliases = migrator.generate_aliases(tag_name, description)
            
            print(f"\n{'='*60}")
            print(f"標籤: {tag_name}")
            print(f"分類: {category}")
            print(f"視覺線索: {visual_cues}")
            print(f"相關標籤: {related_tags}")
            print(f"別名: {aliases}")
    else:
        enhanced_tags = migrator.migrate_tags(args.input, args.output)
        
        report = migrator.generate_report(enhanced_tags)
        
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"報告已保存到: {args.report}")
        print(f"\n{'='*60}")
        print("遷移報告摘要:")
        print(f"總標籤數: {report['total_tags']}")
        print(f"分類統計: {report['categories']}")
        print(f"有視覺線索的標籤: {report['tags_with_visual_cues']} (平均 {report['average_visual_cues']} 個)")
        print(f"有相關標籤的標籤: {report['tags_with_related_tags']} (平均 {report['average_related_tags']} 個)")
        print(f"有別名的標籤: {report['tags_with_aliases']} (平均 {report['average_aliases']} 個)")


if __name__ == "__main__":
    main()
