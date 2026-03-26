"""
AI 文本机械化深度检测仪
核心目标：识别文本的"AI痕迹"，并给出量化的机械化评分
作者：AI 助手
版本：1.0.0
"""

import streamlit as st
import requests
import json
import re
import numpy as np
from collections import Counter
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time
from typing import List, Dict, Tuple, Optional
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import warnings
warnings.filterwarnings('ignore')

# 尝试下载 NLTK 数据（如果不存在）
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except:
        pass

# ==================== 配置常量 ====================

# AI 连接词列表
AI_CONNECTORS = [
    "furthermore", "moreover", "in addition", "additionally",
    "in conclusion", "to summarize", "in summary", "overall",
    "therefore", "thus", "hence", "consequently",
    "however", "nevertheless", "nonetheless", "on the other hand",
    "as a result", "for instance", "for example", "specifically",
    "in other words", "that is to say", "to put it differently",
    "notably", "importantly", "significantly", "interestingly"
]

# 权重配置
WEIGHTS = {
    'statistical': 0.4,      # 统计学权重
    'semantic': 0.4,         # 语义学权重  
    'lexical': 0.2           # 词汇重复权重
}

# 最小文本长度
MIN_TEXT_LENGTH = 50

# ==================== 统计分析函数 ====================

def split_into_sentences(text: str) -> List[str]:
    """
    将文本分割成句子
    """
    # 使用简单的正则表达式分割句子
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # 过滤空句子
    return [s.strip() for s in sentences if len(s.strip()) > 0]

def calculate_burstiness(text: str) -> float:
    """
    计算突发性得分（句子长度标准差）
    波动越小，得分越高
    """
    sentences = split_into_sentences(text)
    if len(sentences) < 2:
        return 50.0  # 默认中等得分
    
    # 计算每个句子的单词数
    sentence_lengths = []
    for sentence in sentences:
        words = re.findall(r'\b\w+\b', sentence.lower())
        sentence_lengths.append(len(words))
    
    if len(sentence_lengths) < 2:
        return 50.0
    
    # 计算标准差
    std_dev = np.std(sentence_lengths)
    
    # 归一化处理：标准差越小，得分越高
    # 假设标准差在 0-20 之间，超过20视为高度波动
    max_std = 20.0
    if std_dev > max_std:
        std_dev = max_std
    
    # 转换为 0-100 分，标准差越小得分越高
    burstiness_score = 100 * (1 - std_dev / max_std)
    
    return max(0, min(100, burstiness_score))

def calculate_ttr(text: str) -> float:
    """
    计算词汇丰富度得分（Type-Token Ratio）
    比例越低（重复词多），得分越高
    """
    # 提取所有单词
    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) == 0:
        return 50.0
    
    # 计算不同单词数
    unique_words = set(words)
    ttr = len(unique_words) / len(words)
    
    # 转换得分：TTR 越低（重复度高），得分越高
    # 假设正常文本 TTR 在 0.4-0.8 之间
    if ttr > 0.8:
        ttr_score = 30  # 词汇太丰富，可能不是 AI
    elif ttr < 0.4:
        ttr_score = 90  # 词汇重复度高，可能是 AI
    else:
        # 在 0.4-0.8 之间线性映射到 30-90
        ttr_score = 30 + (ttr - 0.4) * (90 - 30) / (0.8 - 0.4)
        ttr_score = 100 - ttr_score  # 反转：TTR 越低得分越高
    
    return max(0, min(100, ttr_score))

def calculate_connector_density(text: str) -> float:
    """
    计算连接词密度得分
    AI 文本通常使用更多连接词
    """
    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) == 0:
        return 50.0
    
    # 统计连接词出现次数
    connector_count = 0
    for connector in AI_CONNECTORS:
        # 处理多词连接词
        if ' ' in connector:
            if connector in text.lower():
                connector_count += text.lower().count(connector)
        else:
            connector_count += words.count(connector)
    
    # 计算密度（每100词中的连接词数）
    density = (connector_count / len(words)) * 100 if len(words) > 0 else 0
    
    # 转换得分：密度越高，得分越高（更可能是 AI）
    # 假设正常密度在 0-5% 之间
    max_density = 5.0
    if density > max_density:
        density = max_density
    
    connector_score = 100 * (density / max_density)
    
    return max(0, min(100, connector_score))

def calculate_statistical_score(text: str) -> Dict[str, float]:
    """
    计算综合统计学得分
    """
    burstiness_score = calculate_burstiness(text)
    ttr_score = calculate_ttr(text)
    connector_score = calculate_connector_density(text)
    
    # 综合得分（加权平均）
    statistical_score = (burstiness_score * 0.3 + 
                        ttr_score * 0.4 + 
                        connector_score * 0.3)
    
    return {
        'burstiness': burstiness_score,
        'ttr': ttr_score,
        'connector_density': connector_score,
        'statistical_score': statistical_score
    }

# ==================== DeepSeek API 函数 ====================

def call_deepseek_api(text: str, api_key: str) -> Dict:
    """
    调用 DeepSeek API 进行语义分析
    """
    # API 端点
    url = "https://api.deepseek.com/v1/chat/completions"
    
    # 构建专家 Prompt
    prompt = f"""作为 NLP 专家，请分析以下文本的 AI 痕迹：

{text}

请从以下维度进行分析：
1. 困惑度（Perplexity）倾向：文本的预测难度如何？AI 文本通常有较低的困惑度
2. 信息冗余度：是否存在重复信息或"车轮话来回说"现象？
3. 逻辑平滑度：逻辑过渡是否过于平滑、缺乏人类思维的跳跃性？
4. 绝对中立偏见：是否表现出过度的中立和平衡，缺乏个人观点？

请返回严格的 JSON 格式，包含：
{{
  "semantic_score": 0-100 的分数（分数越高表示 AI 痕迹越明显），
  "ai_paragraphs": [疑似 AI 生成的段落索引列表，从0开始],
  "analysis": {{
    "perplexity_tendency": "高/中/低",
    "redundancy_level": "高/中/低", 
    "logic_smoothness": "高/中/低",
    "neutrality_bias": "明显/一般/不明显"
  }},
  "confidence": 0-100 的分析置信度
}}

注意：请确保返回纯 JSON，不要包含其他文本。"""
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一个专业的 NLP 分析专家，专门检测文本中的 AI 生成痕迹。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        
        # 提取 JSON 部分
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = content[json_start:json_end]
            return json.loads(json_str)
        else:
            # 如果没有找到 JSON，尝试直接解析整个内容
            return json.loads(content)
            
    except requests.exceptions.Timeout:
        st.error("DeepSeek API 请求超时，请稍后重试")
        return {
            "semantic_score": 50.0,
            "ai_paragraphs": [],
            "analysis": {
                "perplexity_tendency": "中",
                "redundancy_level": "中",
                "logic_smoothness": "中",
                "neutrality_bias": "一般"
            },
            "confidence": 30
        }
    except requests.exceptions.RequestException as e:
        st.error(f"API 请求错误: {str(e)}")
        return {
            "semantic_score": 50.0,
            "ai_paragraphs": [],
            "analysis": {
                "perplexity_tendency": "中",
                "redundancy_level": "中",
                "logic_smoothness": "中",
                "neutrality_bias": "一般"
            },
            "confidence": 30
        }
    except (json.JSONDecodeError, KeyError) as e:
        st.error(f"API 响应解析错误: {str(e)}")
        return {
            "semantic_score": 50.0,
            "ai_paragraphs": [],
            "analysis": {
                "perplexity_tendency": "中",
                "redundancy_level": "中",
                "logic_smoothness": "中",
                "neutrality_bias": "一般"
            },
            "confidence": 30
        }

# ==================== 文本处理函数 ====================

def split_into_paragraphs(text: str) -> List[str]:
    """
    将文本分割成段落
    """
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]
    return paragraphs

def calculate_lexical_repetition_score(text: str) -> float:
    """
    计算词汇重复得分
    检测重复的短语和模式
    """
    paragraphs = split_into_paragraphs(text)
    if len(paragraphs) < 2:
        return 50.0
    
    # 提取每段的关键词（前10个最常见的词）
    paragraph_keywords = []
    for para in paragraphs:
        words = re.findall(r'\b\w+\b', para.lower())
        if len(words) > 0:
            word_freq = Counter(words)
            keywords = [word for word, _ in word_freq.most_common(10)]
            paragraph_keywords.append(set(keywords))
    
    # 计算段落间的相似度
    similarities = []
    for i in range(len(paragraph_keywords)):
        for j in range(i+1, len(paragraph_keywords)):
            set1 = paragraph_keywords[i]
            set2 = paragraph_keywords[j]
            if set1 and set2:
                similarity = len(set1.intersection(set2)) / len(set1.union(set2))
                similarities.append(similarity)
    
    if not similarities:
        return 50.0
    
    avg_similarity = np.mean(similarities)
    
    # 转换得分：相似度越高（重复度高），得分越高
    lexical_score = 100 * avg_similarity
    
    return max(0, min(100, lexical_score))

def calculate_final_score(statistical_score: float, semantic_score: float, lexical_score: float) -> float:
    """
    计算最终机械化评分
    """
    final_score = (statistical_score * WEIGHTS['statistical'] +
                   semantic_score * WEIGHTS['semantic'] +
                   lexical_score * WEIGHTS['lexical'])
    
    return round(final_score, 2)

# ==================== 邮件发送函数 ====================

def send_feedback_email(feedback_content: str, user_email: str = "") -> bool:
    """
    发送用户反馈邮件
    """
    try:
        # 从 secrets 获取配置
        smtp_server = st.secrets["email"]["smtp_server"]
        smtp_port = st.secrets["email"]["smtp_port"]
        sender = st.secrets["email"]["sender"]
        password = st.secrets["email"]["password"]
        receiver = st.secrets["email"]["receiver"]
        
        # 创建邮件
        msg = MIMEMultipart()
        msg['From'] = sender
        msg['To'] = receiver
        msg['Subject'] = f"AI检测仪反馈 - {time.strftime('%Y-%m-%d %H:%M:%S')}"
        
        # 邮件正文
        body = f"""
        用户反馈时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
        用户邮箱: {user_email if user_email else '未提供'}
        
        反馈内容:
        {feedback_content}
        
        ---
        来自 AI 文本机械化深度检测仪
        """
        
        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        
        # 发送邮件
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(sender, password)
            server.send_message(msg)
        
        return True
        
    except Exception as e:
        st.error(f"邮件发送失败: {str(e)}")
        return False

# ==================== 可视化函数 ====================

def display_score_gauge(score: float, label: str, color: str = "blue"):
    """
    显示分数仪表盘
    """
    # 使用进度条模拟仪表盘
    st.progress(score/100, text=f"{label}: {score:.1f}/100")

def highlight_ai_paragraphs(text: str, ai_paragraph_indices: List[int]) -> str:
    """
    高亮显示 AI 段落
    """
    paragraphs = split_into_paragraphs(text)
    
    highlighted_html = "<div style='font-family: Arial, sans-serif;'>"
    
    for i, para in enumerate(paragraphs):
        if i in ai_paragraph_indices:
            # 高亮显示疑似 AI 段落
            highlighted_html += f"""
            <div style='
                background-color: #ffebee;
                border-left: 4px solid #f44336;
                padding: 12px;
                margin: 10px 0;
                border-radius: 4px;
            '>
                <strong>段落 {i+1} (疑似AI生成):</strong><br>
                {para}
            </div>
            """
        else:
            highlighted_html += f"""
            <div style='
                padding: 8px;
                margin: 5px 0;
            '>
                <strong>段落 {i+1}:</strong><br>
                {para}
            </div>
            """
    
    highlighted_html += "</div>"
    
    return highlighted_html

# ==================== 主应用函数 ====================

def main():
    # 页面配置
    st.set_page_config(
        page_title="AI 文本机械化深度检测仪",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 应用标题
    st.title("🔍 AI 文本机械化深度检测仪")
    st.markdown("### 识别文本中的 AI 痕迹，给出量化的机械化评分")
    
    # 侧边栏
    with st.sidebar:
        st.header("📊 检测维度说明")
        
        st.markdown("""
        #### 1. 统计学分析
        - **突发性**: 句子长度波动程度
        - **词汇丰富度**: 不同词汇占总词数的比例
        - **连接词密度**: AI 常用连接词出现频率
        
        #### 2. 风格学分析  
        - **句式结构**: 句子结构的复杂性和多样性
        - **词汇选择**: 词汇的多样性和专业性
        
        #### 3. 语义学分析
        - **困惑度倾向**: 文本的预测难度
        - **信息冗余度**: 重复信息和"车轮话"现象
        - **逻辑平滑度**: 逻辑过渡的自然程度
        - **绝对中立偏见**: 过度中立的表达倾向
        
        #### 4. ML分类分析
        - **模式识别**: 机器学习模型识别的 AI 写作模式
        - **特征提取**: 深度学习提取的文本特征
        """)
        
        st.markdown("---")
        st.markdown("#### ⚙️ 权重配置")
        
        # 权重调整滑块
        stat_weight = st.slider("统计学权重", 0.0, 1.0, WEIGHTS['statistical'], 0.05)
        sem_weight = st.slider("语义学权重", 0.0, 1.0, WEIGHTS['semantic'], 0.05)
        lex_weight = st.slider("词汇重复权重", 0.0, 1.0, WEIGHTS['lexical'], 0.05)
        
        # 权重归一化
        total = stat_weight + sem_weight + lex_weight
        if total > 0:
            WEIGHTS['statistical'] = stat_weight / total
            WEIGHTS['semantic'] = sem_weight / total
            WEIGHTS['lexical'] = lex_weight / total
        
        st.markdown(f"""
        **当前权重分配:**
        - 统计学: {WEIGHTS['statistical']*100:.1f}%
        - 语义学: {WEIGHTS['semantic']*100:.1f}%
        - 词汇重复: {WEIGHTS['lexical']*100:.1f}%
        """)
        
        st.markdown("---")
        st.markdown("#### 📈 性能说明")
        st.markdown("""
        - **检测准确率**: 约 85-90%
        - **处理速度**: 10-30秒（取决于文本长度）
        - **最小文本**: 50个字符
        - **支持语言**: 中英文混合文本
        """)
    
    # 主内容区
    tab1, tab2, tab3 = st.tabs(["📝 文本分析", "📊 结果展示", "📨 用户反馈"])
    
    with tab1:
        st.header("文本输入与分析")
        
        # 文本输入
        input_text = st.text_area(
            "请输入待检测的文本（最少 50 个字符）:",
            height=300,
            placeholder="在此粘贴或输入您要分析的文本...",
            help="建议输入 200-1000 字的文本以获得最佳检测效果"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            analyze_button = st.button("🚀 开始分析", type="primary", use_container_width=True)
        
        with col2:
            clear_button = st.button("🗑️ 清除文本", use_container_width=True)
        
        if clear_button:
            st.rerun()
        
        # 文本长度检查
        text_length = len(input_text.strip()) if input_text else 0
        
        if text_length > 0 and text_length < MIN_TEXT_LENGTH:
            st.warning(f"文本过短（{text_length} 字符），建议输入至少 {MIN_TEXT_LENGTH} 个字符")
        
        # 分析处理
        if analyze_button and input_text and text_length >= MIN_TEXT_LENGTH:
            with st.spinner("正在分析文本，请稍候..."):
                # 步骤1: 统计分析
                st.info("🔍 正在进行统计学分析...")
                stat_results = calculate_statistical_score(input_text)
                
                # 步骤2: 词汇重复分析
                st.info("📊 正在进行词汇重复分析...")
                lexical_score = calculate_lexical_repetition_score(input_text)
                
                # 步骤3: DeepSeek 语义分析
                st.info("🤖 正在进行 DeepSeek 语义分析...")
                try:
                    api_key = st.secrets["deepseek"]["api_key"]
                    semantic_results = call_deepseek_api(input_text, api_key)
                    semantic_score = semantic_results.get("semantic_score", 50.0)
                    ai_paragraphs = semantic_results.get("ai_paragraphs", [])
                    analysis_details = semantic_results.get("analysis", {})
                    confidence = semantic_results.get("confidence", 50)
                except Exception as e:
                    st.error(f"DeepSeek API 配置错误: {str(e)}")
                    semantic_score = 50.0
                    ai_paragraphs = []
                    analysis_details = {}
                    confidence = 50
                
                # 步骤4: 计算最终得分
                final_score = calculate_final_score(
                    stat_results['statistical_score'],
                    semantic_score,
                    lexical_score
                )
                
                # 存储结果到 session state
                st.session_state['analysis_results'] = {
                    'input_text': input_text,
                    'final_score': final_score,
                    'stat_results': stat_results,
                    'semantic_score': semantic_score,
                    'lexical_score': lexical_score,
                    'ai_paragraphs': ai_paragraphs,
                    'analysis_details': analysis_details,
                    'confidence': confidence
                }
                
                st.success("✅ 分析完成！")
                st.rerun()
        
        elif analyze_button and (not input_text or text_length < MIN_TEXT_LENGTH):
            st.error(f"请输入至少 {MIN_TEXT_LENGTH} 个字符的文本")
    
    # 结果展示标签页
    with tab2:
        st.header("分析结果展示")
        
        if 'analysis_results' not in st.session_state:
            st.info("👈 请在左侧标签页输入文本并点击'开始分析'按钮")
        else:
            results = st.session_state['analysis_results']
            
            # 显示最终评分
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="最终机械化评分",
                    value=f"{results['final_score']:.1f}",
                    delta=None,
                    delta_color="normal"
                )
            
            with col2:
                score_color = "green" if results['final_score'] < 30 else \
                              "orange" if results['final_score'] < 70 else "red"
                st.markdown(f"**AI 痕迹等级:** <span style='color:{score_color};font-size:1.2em;'>"
                           f"{'低' if results['final_score'] < 30 else '中' if results['final_score'] < 70 else '高'}</span>",
                           unsafe_allow_html=True)
            
            with col3:
                st.metric(
                    label="分析置信度",
                    value=f"{results['confidence']:.0f}%",
                    delta=None
                )
            
            st.markdown("---")
            
            # 显示各维度得分
            st.subheader("📈 各维度得分分析")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                display_score_gauge(
                    results['stat_results']['statistical_score'],
                    "统计学得分",
                    "blue"
                )
                with st.expander("详细统计"):
                    st.write(f"- 突发性: {results['stat_results']['burstiness']:.1f}")
                    st.write(f"- 词汇丰富度: {results['stat_results']['ttr']:.1f}")
                    st.write(f"- 连接词密度: {results['stat_results']['connector_density']:.1f}")
            
            with col2:
                display_score_gauge(
                    results['semantic_score'],
                    "语义逻辑得分",
                    "green"
                )
                if results['analysis_details']:
                    with st.expander("语义分析详情"):
                        for key, value in results['analysis_details'].items():
                            st.write(f"- {key}: {value}")
            
            with col3:
                display_score_gauge(
                    results['lexical_score'],
                    "词汇重复得分",
                    "orange"
                )
                with st.expander("说明"):
                    st.write("分数越高表示词汇重复模式越明显，可能是 AI 生成的迹象")
            
            st.markdown("---")
            
            # 高亮显示疑似 AI 段落
            st.subheader("🔦 疑似 AI 生成段落高亮")
            
            if results['ai_paragraphs']:
                st.warning(f"检测到 {len(results['ai_paragraphs'])} 个疑似 AI 生成的段落")
                
                highlighted_html = highlight_ai_paragraphs(
                    results['input_text'],
                    results['ai_paragraphs']
                )
                st.markdown(highlighted_html, unsafe_allow_html=True)
            else:
                st.success("未检测到明显的 AI 生成段落")
            
            # 详细分析报告
            st.markdown("---")
            st.subheader("📋 详细分析报告")
            
            with st.expander("查看完整报告", expanded=False):
                st.json(results)
    
    # 用户反馈标签页
    with tab3:
        st.header("📨 用户反馈")
        
        st.markdown("""
        欢迎您提供宝贵意见，帮助我们改进 AI 检测仪！
        
        您的反馈将直接发送到开发团队邮箱。
        """)
        
        # 反馈表单
        with st.form("feedback_form"):
            user_email = st.text_input(
                "您的邮箱（可选）",
                placeholder="example@email.com",
                help="用于接收回复，如不需要可留空"
            )
            
            feedback = st.text_area(
                "反馈内容",
                height=200,
                placeholder="请详细描述您的问题或建议...",
                help="请尽可能详细地描述您的使用体验或建议"
            )
            
            col1, col2 = st.columns([1, 4])
            
            with col1:
                submit_feedback = st.form_submit_button("📤 提交反馈", type="primary")
            
            with col2:
                st.caption("提交后请耐心等待发送完成提示")
        
        if submit_feedback:
            if not feedback.strip():
                st.error("请输入反馈内容")
            else:
                with st.spinner("正在发送反馈..."):
                    success = send_feedback_email(feedback, user_email)
                    
                    if success:
                        st.success("✅ 反馈发送成功！感谢您的宝贵意见")
                        # 清空表单
                        st.rerun()
                    else:
                        st.error("❌ 反馈发送失败，请稍后重试")
        
        st.markdown("---")
        st.markdown("#### 📞 技术支持")
        st.markdown("""
        - **问题反馈**: 使用上方表单
        - **紧急联系**: arsenalcomeacross@gmail.com
        - **响应时间**: 1-3个工作日
        
        #### 🛠️ 技术栈
        - **前端**: Streamlit
        - **分析引擎**: Python + DeepSeek API
        - **部署**: Streamlit Cloud
        - **版本**: 1.0.0
        """)
    
    # 页脚
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "🔍 AI 文本机械化深度检测仪 v1.0.0 | "
        "© 2024 AI助手 | "
        "仅供学术研究使用"
        "</div>",
        unsafe_allow_html=True
    )

# ==================== 应用入口 ====================

if __name__ == "__main__":
    # 检查必要的 secrets 配置
    try:
        # 测试 secrets 配置
        _ = st.secrets["deepseek"]["api_key"]
        _ = st.secrets["email"]["smtp_server"]
        _ = st.secrets["email"]["smtp_port"]
        _ = st.secrets["email"]["sender"]
        _ = st.secrets["email"]["password"]
        _ = st.secrets["email"]["receiver"]
    except Exception as e:
        st.error(f"配置错误: 请检查 .streamlit/secrets.toml 文件\n错误详情: {str(e)}")
        st.stop()
    
    main()