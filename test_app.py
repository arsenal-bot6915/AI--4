"""
测试 AI 文本机械化深度检测仪的核心功能
"""

import sys
import os

# 添加当前目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入 app 中的函数（不运行 Streamlit）
try:
    # 测试文本
    test_text = """
    Artificial intelligence has revolutionized many industries in recent years. 
    Furthermore, it has transformed the way we approach problem-solving. 
    In conclusion, AI will continue to shape our future. Moreover, machine learning 
    algorithms are becoming increasingly sophisticated. Therefore, we must adapt 
    to these changes. However, there are also challenges to consider.
    
    The impact of artificial intelligence is profound. Furthermore, it affects 
    various sectors including healthcare, finance, and education. In summary, 
    AI technologies offer numerous benefits. Additionally, they present new 
    opportunities for innovation. Consequently, organizations are investing 
    heavily in AI research.
    """
    
    print("测试文本长度:", len(test_text), "字符")
    print("=" * 60)
    
    # 测试句子分割
    from app import split_into_sentences
    sentences = split_into_sentences(test_text)
    print("句子数量:", len(sentences))
    print("前3个句子:", sentences[:3])
    print("=" * 60)
    
    # 测试段落分割
    from app import split_into_paragraphs
    paragraphs = split_into_paragraphs(test_text)
    print("段落数量:", len(paragraphs))
    print("=" * 60)
    
    # 测试突发性计算
    from app import calculate_burstiness
    burstiness_score = calculate_burstiness(test_text)
    print("突发性得分:", burstiness_score)
    print("=" * 60)
    
    # 测试词汇丰富度
    from app import calculate_ttr
    ttr_score = calculate_ttr(test_text)
    print("词汇丰富度得分:", ttr_score)
    print("=" * 60)
    
    # 测试连接词密度
    from app import calculate_connector_density
    connector_score = calculate_connector_density(test_text)
    print("连接词密度得分:", connector_score)
    print("=" * 60)
    
    # 测试综合统计学得分
    from app import calculate_statistical_score
    stat_results = calculate_statistical_score(test_text)
    print("综合统计学得分:", stat_results['statistical_score'])
    print("详细统计:", stat_results)
    print("=" * 60)
    
    # 测试词汇重复得分
    from app import calculate_lexical_repetition_score
    lexical_score = calculate_lexical_repetition_score(test_text)
    print("词汇重复得分:", lexical_score)
    print("=" * 60)
    
    # 测试最终得分计算
    from app import calculate_final_score, WEIGHTS
    final_score = calculate_final_score(
        stat_results['statistical_score'],
        65.0,  # 模拟语义得分
        lexical_score
    )
    print("权重配置:", WEIGHTS)
    print("最终机械化评分（模拟）:", final_score)
    print("=" * 60)
    
    print("✅ 所有核心函数测试通过！")
    
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()