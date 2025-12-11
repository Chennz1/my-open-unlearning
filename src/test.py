import hydra
import json
import pprint
from omegaconf import DictConfig, OmegaConf
from data import get_data, get_collators
from model import get_model,get_tokenizer
from trainer import load_trainer
from evals import get_evaluators
from trainer.utils import seed_everything


def pretty_print_dict(data, title="数据信息", style="pprint", max_width=120):
    """
    格式化打印字典或对象信息
    
    Args:
        data: 要打印的数据
        title: 标题
        style: 打印风格 ('pprint', 'json', 'yaml')
        max_width: 最大宽度
    """
    print("=" * max_width)
    print(f"{title}:")
    print("=" * max_width)
    
    if style == "json":
        # 使用JSON格式打印
        try:
            if isinstance(data, DictConfig):
                # OmegaConf对象转换为字典
                data_dict = OmegaConf.to_container(data, resolve=True)
                print(json.dumps(data_dict, indent=2, ensure_ascii=False, default=str))
            elif hasattr(data, '__dict__'):
                print(json.dumps(vars(data), indent=2, ensure_ascii=False, default=str))
            else:
                print(json.dumps(data, indent=2, ensure_ascii=False, default=str))
        except (TypeError, ValueError):
            print("无法序列化为JSON，使用pprint格式:")
            pprint.pprint(data, width=max_width, depth=4)
    
    elif style == "yaml" and isinstance(data, DictConfig):
        # 使用YAML格式打印（仅适用于OmegaConf）
        print(OmegaConf.to_yaml(data))
    
    else:
        # 使用pprint格式打印
        if hasattr(data, '__dict__') and not isinstance(data, (dict, list, tuple)):
            pprint.pprint(vars(data), width=max_width, depth=4)
        else:
            pprint.pprint(data, width=max_width, depth=4)
    
    print("=" * max_width)


def print_dataset_info(data, title="数据集信息"):
    """
    专门用于打印数据集信息的函数
    """
    print("=" * 120)
    print(f"{title}:")
    print("=" * 120)
    
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"🔹 数据集类型: {key}")
            
            # 打印基本信息
            if hasattr(value, '__len__'):
                print(f"   📊 数据量: {len(value)}")
            if hasattr(value, '__class__'):
                print(f"   🏷️  数据类型: {value.__class__.__name__}")
            
            # 打印详细信息
            print("   📋 详细信息:")
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    print(f"      {sub_key}: {type(sub_value).__name__}")
                    if isinstance(sub_value, (list, tuple)) and len(sub_value) > 0:
                        print(f"         样例: {str(sub_value[0])[:100]}...")
            elif hasattr(value, '__dict__'):
                important_attrs = {k: v for k, v in vars(value).items() 
                                 if not k.startswith('_') and not callable(v)}
                for attr, val in list(important_attrs.items())[:5]:  # 只显示前5个属性
                    print(f"      {attr}: {type(val).__name__}")
            
            print("-" * 60)
    else:
        pprint.pprint(data, width=120, depth=3)
    
    print("=" * 120)


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to train models
    Args:
        cfg (DictConfig): Config to train
    """
    seed_everything(cfg.trainer.args.seed)
    mode = cfg.get("mode", "train")
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    
    # 格式化打印配置信息
    pretty_print_dict(cfg.data, title="配置信息 (Configuration)", style="yaml")
    pretty_print_dict(cfg.collator, title="配置信息 (Configuration)", style="json")
    
    model, tokenizer = get_model(model_cfg)
    tokenizer_args = model_cfg.tokenizer_args
    tokenizer = get_tokenizer(tokenizer_args)

    # Load Dataset
    data_cfg = cfg.data
    data = get_data(
        data_cfg, mode=mode, tokenizer=tokenizer, template_args=template_args
    )
    # print(data['retain'])
    # print(data['forget'])
    print(data.get("train", None))
    # Load collator
    collator_cfg = cfg.collator
    collator = get_collators(collator_cfg, tokenizer=tokenizer)
    
    # 格式化打印 collators 信息
    # pretty_print_dict(collator, title="数据整理器 (Data Collators)", style="pprint")
    
    # 格式化打印 data 信息
    # print_dataset_info(data)
    
    # Get Trainer
    trainer_cfg = cfg.trainer
    assert trainer_cfg is not None, ValueError("Please set trainer")

    # Get Evaluators
    evaluators = None
    eval_cfgs = cfg.get("eval", None)
    if eval_cfgs:
        evaluators = get_evaluators(
            eval_cfgs=eval_cfgs,
            template_args=template_args,
            model=model,
            tokenizer=tokenizer,
        )

    trainer, trainer_args = load_trainer(
        trainer_cfg=trainer_cfg,
        model=model,
        train_dataset=data.get("train", None),
        eval_dataset=data.get("eval", None),
        tokenizer=tokenizer,
        data_collator=collator,
        evaluators=evaluators,
        template_args=template_args,
    )
    print(trainer_args)

    # if trainer_args.do_train:
    #     trainer.train()
    #     trainer.save_state()
    #     trainer.save_model(trainer_args.output_dir)

    # if trainer_args.do_eval:
    #     trainer.evaluate(metric_key_prefix="eval")


if __name__ == "__main__":
    main()
