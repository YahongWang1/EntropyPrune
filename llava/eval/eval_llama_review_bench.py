import argparse
import json
import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 配置本地模型路径 ---
MODEL_PATH = "/wyh/LVLM/Model/Model_weights/llm/Llama-3.1-8B-Instruct"

def load_local_model():
    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    # 使用 device_map="auto" 自动分配 GPU，torch_dtype 设为 bfloat16 (Llama3 推荐) 或 float16
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    )
    print("Model loaded successfully.")
    return model, tokenizer

def get_eval_local(model, tokenizer, content, max_tokens):
    # 构造符合 Llama-3 格式的消息列表
    # 保持原代码逻辑：System Prompt 设置为裁判角色，User 内容为具体的上下文和回答
    messages = [
        {
            'role': 'system',
            'content': 'You are a helpful and precise assistant for checking the quality of the answer.'
        },
        {
            'role': 'user',
            'content': content,
        }
    ]

    # 应用 Chat Template (这会自动处理 Llama-3 特有的特殊 token)
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    # Llama-3 特有的停止符设置
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    # 生成回复
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.2, # 保持原代码的 temperature
            top_p=0.9,
        )
    
    # 解码输出 (只取生成的部分，去掉 prompt)
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)


def parse_score(review):
    try:
        score_pair = review.split('\n')[0]
        score_pair = score_pair.replace(',', ' ')
        sp = score_pair.split(' ')
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            print('error', review)
            return [-1, -1]
    except Exception as e:
        print(e)
        print('error', review)
        return [-1, -1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Local LLM-based QA evaluation.')
    parser.add_argument('-q', '--question', required=True)
    parser.add_argument('-c', '--context', required=True)
    parser.add_argument('-a', '--answer-list', nargs='+', default=[], required=True)
    parser.add_argument('-r', '--rule', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('--max-tokens', type=int, default=1024, help='maximum number of tokens produced in the output')
    args = parser.parse_args()

    # --- 1. 在循环开始前加载模型 ---
    model, tokenizer = load_local_model()

    f_q = open(os.path.expanduser(args.question))
    f_ans1 = open(os.path.expanduser(args.answer_list[0]))
    f_ans2 = open(os.path.expanduser(args.answer_list[1]))
    rule_dict = json.load(open(os.path.expanduser(args.rule), 'r'))

    if os.path.isfile(os.path.expanduser(args.output)):
        cur_reviews = [json.loads(line) for line in open(os.path.expanduser(args.output))]
    else:
        cur_reviews = []

    # 使用 'a' 模式追加，建议设置 buffering=1 确保每行都写入，防止程序中断数据丢失
    review_file = open(f'{args.output}', 'a', buffering=1)

    context_list = [json.loads(line) for line in open(os.path.expanduser(args.context))]
    image_to_context = {context['image']: context for context in context_list}

    idx = 0
    # 遍历数据
    for ques_js, ans1_js, ans2_js in zip(f_q, f_ans1, f_ans2):
        # 如果已经评测过，跳过
        if idx < len(cur_reviews):
            idx += 1
            continue

        ques = json.loads(ques_js)
        ans1 = json.loads(ans1_js)
        ans2 = json.loads(ans2_js)

        inst = image_to_context[ques['image']]

        if isinstance(inst['caption'], list):
            cap_str = '\n'.join(inst['caption'])
        else:
            cap_str = inst['caption']

        category = 'llava_bench_' + json.loads(ques_js)['category']
        if category in rule_dict:
            rule = rule_dict[category]
        else:
            assert False, f"Visual QA category not found in rule file: {category}."
        
        prompt = rule['prompt']
        role = rule['role']
        content = (f'[Context]\n{cap_str}\n\n'
                   f'[Question]\n{ques["text"]}\n\n'
                   f'[{role} 1]\n{ans1["text"]}\n\n[End of {role} 1]\n\n'
                   f'[{role} 2]\n{ans2["text"]}\n\n[End of {role} 2]\n\n'
                   f'[System]\n{prompt}\n\n')
        
        cur_js = {
            'id': idx+1,
            'question_id': ques['question_id'],
            'answer1_id': ans1.get('answer_id', ans1['question_id']),
            'answer2_id': ans2.get('answer_id', ans2['answer_id']),
            'category': category
        }

        # --- 2. 调用本地模型生成 ---
        try:
            print(f"Evaluating ID: {idx+1}")
            review = get_eval_local(model, tokenizer, content, args.max_tokens)
            scores = parse_score(review)
            cur_js['content'] = review
            cur_js['tuple'] = scores
            
            review_file.write(json.dumps(cur_js) + '\n')
            # review_file.flush() # 由于 open 时设置了 buffering=1，这里其实不需要手动 flush 了，但保留也无妨
        except Exception as e:
            print(f"Error processing ID {idx+1}: {e}")
        
        idx += 1

    review_file.close()
    print("Evaluation finished.")