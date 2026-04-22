"""Business logic constants extracted from Settings class.

These constants are used throughout the application for tag matching,
confidence scoring, and RAG operations.
"""

# Sensitive tags requiring visual verification
SENSITIVE_TAGS: frozenset[str] = frozenset(
    {
        # English sensitive tags
        "loli",
        "shota",
        "anal",
        "rape",
        "incest",
        "tentacles",
        "guro",
        "vore",
        "pedophilia",
        "child",
        "infant",
        "baby",
        "lolicon",
        "shotacon",
        "bestiality",
        " zoophilia",
        "futanari",
        "shemale",
        "trap",
        "crossdresser",
        "prostitution",
        "escort",
        "sex",
        "oral",
        "blowjob",
        "paizuri",
        "titfuck",
        "handjob",
        "finger",
        "cunnilingus",
        "rimming",
        "footjob",
        "masturbation",
        "bondage",
        "bdsm",
        "domination",
        "submission",
        "sadism",
        "masochism",
        "cum",
        "ejaculation",
        "orgasm",
        "squirting",
        "creampie",
        "facial",
        "assjob",
        "spanking",
        "wrestling",
        "fighting",
        "threesome",
        "group",
        "orgy",
        "gangbang",
        "double penetration",
        "triple penetration",
        "rape",
        "non-consensual",
        "coercion",
        "force",
        "mind break",
        "brainwashing",
        "hypnosis",
        "mind control",
        "possession",
        "body swap",
        "transformation",
        "gender change",
        "gender bender",
        # Chinese sensitive tags
        "蘿莉",
        "正太",
        "低含量蘿莉",
        "蘿莉塔",
        "幼女",
        "小女孩",
        "肛交",
        "口交",
        "乳交",
        "手淫",
        "自慰",
        "足交",
        "舔腳",
        "舔屁眼",
        "舔陰蒂",
        "肛門插入",
        "尿道插入",
        "綁縛",
        "調教",
        "奴隸",
        "凌虐",
        "強制",
        "強姦",
        "輪姦",
        "亂倫",
        "低含量亂倫",
        "近親",
        "母女",
        "姐妹",
        "父子",
        "母子",
        "獸交",
        "低含量獸交",
        "觸手",
        "獵奇",
        "食人",
        "斷肢",
        "血腥",
        "殘忍",
        "墮落",
        "洗腦",
        "催眠",
        "附身",
        "換身",
        "變身",
        "性轉換",
        "扶他",
        "人妖",
        "偽娘",
        "賣淫",
        "出軌",
        "NTR",
        "綠帽癖",
        "換妻",
        "鬼畜",
        "精神",
        "凌辱",
        "調教",
        "奴隸",
        "人寵",
        "人肉按摩棒",
        "顏射",
        "內射",
        "潮吹",
        "受精",
        "懷孕",
        "產卵",
        "打屁股",
        "拳交",
        "多重插入",
        "兩屌一洞",
        "三屌一洞",
        "人肉按摩棒",
        "電擊",
        "毆打",
        "拷打",
        "踩踏",
        "鞭打",
        "絞刑",
        "殺害",
    }
)

# Tag frequency calibration factors - adjusts confidence based on tag prevalence
TAG_FREQUENCY_CALIBRATION: dict[str, float] = {
    # Core character types
    "巨乳": 0.90,
    "爆乳": 0.88,
    "貧乳": 0.92,
    "普通乳": 1.0,
    "蘿莉": 0.88,
    "正太": 0.90,
    "少女": 0.85,
    "少年": 0.85,
    "人妻": 1.03,
    "熟女": 1.02,
    "御姐": 1.02,
    # Animal-eared characters
    "獸耳": 1.05,
    "貓娘": 1.02,
    "狐娘": 1.02,
    "犬娘": 1.02,
    "狐狸娘": 1.02,
    # Fantasy characters
    "精靈": 1.05,
    "天使": 1.05,
    "惡魔娘": 1.05,
    "吸血鬼": 1.05,
    "人魚": 1.05,
    "魔物娘": 1.03,
    "龍娘": 1.03,
    "仙女": 1.03,
    "幽靈": 1.03,
    "機器人": 1.03,
    "機娘": 1.03,
    # Hair features
    "雙馬尾": 1.03,
    "馬尾": 1.01,
    "單馬尾": 1.01,
    "長髮": 1.01,
    "短髮": 1.01,
    "捲髮": 1.01,
    "丸子頭": 1.02,
    "雙馬尾": 1.03,
    # Clothing
    "女生制服": 0.95,
    "男生制服": 0.95,
    "泳裝": 0.95,
    "比基尼": 0.96,
    "情趣內衣": 0.92,
    "和服": 1.02,
    "浴衣": 1.02,
    "女僕裝": 1.02,
    "護士裝": 1.02,
    "兔女郎": 1.01,
    "巫女裝": 1.02,
    "旗袍": 1.02,
    "運動服": 1.01,
    "婚紗": 1.02,
    "西裝": 1.02,
    "啦啦隊員": 1.02,
    "女忍裝": 1.02,
    "白袍": 1.02,
    "軍裝": 1.02,
    # Accessories
    "眼鏡": 1.02,
    "隱形眼鏡": 1.02,
    "翅膀": 1.03,
    "尾巴": 1.03,
    "角": 1.03,
    "光環": 1.03,
    # Body features
    "紋身": 1.02,
    "肌肉": 1.02,
    "大肌肉": 1.02,
    # Themes
    "百合": 1.02,
    "耽美": 1.02,
    "NTR": 0.90,
    "後宮": 1.01,
    "純愛": 1.03,
    "學園": 1.02,
    "奴隸": 0.95,
    "調教": 0.92,
    "凌虐": 0.90,
    "戰鬥": 1.01,
    "奇幻": 1.02,
    "科幻": 1.02,
    "恐怖": 0.95,
    "搞笑": 1.01,
    "溫馨": 1.02,
    "運動": 1.01,
    "音樂": 1.01,
    # Art styles
    "黑白": 1.01,
    "彩色": 1.01,
    "草圖": 1.02,
    "線稿": 1.02,
    "完稿": 1.01,
    "厚塗": 1.01,
    # Actions
    "肛交": 0.85,
    "觸手": 0.85,
    "口交": 0.88,
    "乳交": 0.88,
    "自慰": 0.85,
    "綁縛": 0.90,
    "強姦": 0.80,
    "輪姦": 0.80,
    # Additional anime-specific
    "魔法少女": 1.03,
    "異世界": 1.02,
    "轉生": 1.02,
    # Additional body features
    "大乳暈": 0.92,
    "乳頭內凹": 0.92,
    "多乳頭": 0.90,
    "曬痕": 0.92,
    "雀斑": 0.95,
    "疤痕": 0.92,
    # Hair colors
    "金髮": 1.01,
    "紅髮": 1.01,
    "藍髮": 1.01,
    "白髮": 1.01,
    "黑髮": 1.01,
    "紫髮": 1.01,
    "綠髮": 1.01,
    "銀髮": 1.01,
    # Additional clothing
    "連身裙": 1.01,
    "熱褲": 0.98,
    "內衣": 0.95,
    "緊身衣": 0.98,
    "乳膠緊身衣": 0.95,
    "高跟鞋": 1.01,
    "高筒靴": 1.01,
    "絲襪": 1.02,
    "長筒襪": 1.02,
    "過膝襪": 1.02,
    "連褲襪": 1.02,
    # Additional character types
    "偽娘": 0.90,
    "扶他": 0.90,
    "人妖": 0.90,
    # Additional themes
    "後宮": 1.01,
    "逆後宮": 1.01,
    "出軌": 0.90,
    "換妻": 0.88,
    "綠帽癖": 0.88,
    "亂倫": 0.85,
    "百合": 1.02,
    "耽美": 1.02,
}

# Semantic sibling tags - related tags that should be considered together
SEMANTIC_SIBLINGS: dict[str, set[str]] = {
    "女生制服": {"男生制服"},
    "泳裝": {"比基尼"},
    "巨乳": {"貧乳"},
    "調教": {"奴隸"},
}

# Threshold for considering tags as semantic siblings
SEMANTIC_SIBLING_THRESHOLD: float = 0.85

# Match boosting factors for tag confidence calculation
EXACT_MATCH_BOOST: float = 1.1
PARTIAL_MATCH_BOOST: float = 1.0
SEMANTIC_MATCH_PENALTY: float = 0.95

# RAG support boosting and decay factors
RAG_SUPPORT_BOOST: float = 1.15  # Increased from 1.05 to 1.15
RAG_SUPPORT_DECAY: float = 0.85  # Increased from 0.95 to 0.85 (more penalty for no RAG)

# Minimum acceptable confidence for tag inclusion
MIN_ACCEPTABLE_CONFIDENCE: float = 0.55  # Increased from 0.45 to 0.55

# RAG similarity threshold for considering matches
RAG_SIMILARITY_THRESHOLD: float = 0.60

# Exact match penalty for common false positives
EXACT_MATCH_PENALTY: dict[str, float] = {
    "蘿莉": 0.75,  # Often over-detected
    "少女": 0.85,  # Too generic
    "女生": 0.80,  # Too generic
    "巨乳": 0.90,  # Often false positive
    "貧乳": 0.90,  # Often false positive
    "獸耳": 0.85,  # Needs visual verification
    "女僕裝": 0.90,  # Sometimes confused
    "泳裝": 0.90,  # Sometimes confused
    "比基尼": 0.90,  # Sometimes confused
}

# Mutual exclusivity groups (can't have both tags in each group)
MUTUAL_EXCLUSIVITY: dict[str, set[str]] = {
    # Age-related mutually exclusive
    "蘿莉": {"少女", "人妻", "御姐", "熟女", "人妻", "太太", "成年", "大人", "正太"},
    "正太": {"少女", "蘿莉", "少年", "人夫", "大人"},
    "少女": {"蘿莉", "正太", "人妻", "熟女", "御姐"},
    "少年": {"正太", "人夫", "大人"},
    "人妻": {"蘿莉", "少女", "正太", "少年", "未婚"},
    "熟女": {"蘿莉", "少女", "正太"},
    "御姐": {"蘿莉", "少女"},
    # Breast size mutually exclusive
    "巨乳": {"貧乳", "平胸", "普通乳", "爆乳"},
    "爆乳": {"貧乳", "平胸", "普通乳", "巨乳"},
    "貧乳": {"巨乳", "爆乳", "大胸部"},
    "普通乳": {"巨乳", "爆乳", "貧乳"},
    # Theme mutually exclusive
    "百合": {"耽美", "BL", "yaoi"},
    "耽美": {"百合", "GL", "yuri"},
    "純愛": {"NTR", "出軌", "強姦", "亂倫"},
    "NTR": {"純愛", "百合", "耽美"},
    "後宮": {"百合", "耽美", "逆後宮"},
    "逆後宮": {"後宮"},
    # Content mutually exclusive
    "同性戀": {"異性戀", "雙性戀"},
    "百合": {"耽美"},
    "耽美": {"百合"},
    # Art style mutually exclusive
    "黑白": {"彩色"},
    "彩色": {"黑白"},
    # Gender-related mutually exclusive (for characters)
    "偽娘": {"女性", "女生", "女子", "少女", "人妻", "熟女", "御姐"},
    "扶他": {"女性", "男生"},
    "人妖": {"女性", "男生"},
    # Body type mutually exclusive
    "肌肉": {"肥胖", "皮包骨"},
    "大肌肉": {"肥胖", "皮包骨"},
    "棉花糖女孩": {"皮包骨", "肌肉", "大肌肉"},
    "皮包骨": {"棉花糖女孩", "肌肉", "大肌肉"},
    # Clothing mutually exclusive (can't wear both)
    "泳裝": {"和服", "浴衣", "女僕裝", "護士裝", "巫女裝", "婚紗", "西裝"},
    "比基尼": {"連身裙", "女僕裝", "護士裝"},
    "情趣內衣": {"外套", "大衣"},
    "和服": {"泳裝", "比基尼", "女僕裝", "西裝"},
    "女僕裝": {"和服", "浴衣", "巫女裝", "比基尼"},
    # Hair mutually exclusive
    "長髮": {"短髮", "禿頭", "光頭"},
    "短髮": {"長髮", "超長髮"},
    "雙馬尾": {"馬尾", "丸子頭"},
    "馬尾": {"雙馬尾"},
    # More character types
    "動物": {"人類", "精靈", "天使", "惡魔娘"},
    "人類": {"動物", "機器人", "魔物娘"},
}

# Hierarchical tag relationships (specific -> generic mapping)
TAG_HIERARCHY: dict[str, str] = {
    # Animal ears ->兽耳 (specific to generic)
    "貓娘": "獸耳",
    "狐娘": "獸耳",
    "犬娘": "獸耳",
    "狐狸娘": "獸耳",
    "犬耳": "獸耳",
    "狐耳": "獸耳",
    "貓耳": "獸耳",
    # Character age groups
    "蘿莉": "少女",
    "正太": "少年",
    # Clothing hierarchy
    "比基尼": "泳裝",
    "情趣內衣": "內衣",
}

# Visual feature support - tags that benefit from visual confirmation
VISUAL_FEATURE_BOOST: dict[str, float] = {
    "獸耳": 1.08,  # Needs visual cat/dog/fox ears
    "精靈": 1.08,  # Needs visual pointy ears
    "天使": 1.05,  # Needs visual wings/halo
    "惡魔娘": 1.05,  # Needs visual horns/wings
    "雙馬尾": 1.05,  # Needs visual confirmation
    "馬尾": 1.03,
    "眼鏡": 1.05,  # Needs visual glasses
    "女僕裝": 1.05,  # Needs visual confirmation
    "和服": 1.05,  # Needs visual confirmation
    "巫女服": 1.05,  # Needs visual confirmation
    "護士裝": 1.05,  # Needs visual confirmation
    "兔女郎": 1.05,  # Needs visual confirmation
}
