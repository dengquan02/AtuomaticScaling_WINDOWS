```mermaid
graph TD
    subgraph 数据准备
        raw[(GRM/SAO 原始数据)];
        conv[/convert_GRM_to_input<br/>Matlab & Notebook 工具/];
        ds[(dataset/*<br/>train.lst,test.lst,etc.)];
        raw --> conv;
        conv --> ds;
    end

    cfg[实验配置<br/>example_config.yaml / my_config.yaml];
    cli[dias_main.py<br/>命令行入口];
    cfg --> cli;

    cli -- train --> trainer[train(cfgs)];
    cli -- test --> tester[test(cfgs)];
    cli -- eval --> evaluator[eval(cfgs)];

    subgraph 训练流程
        trainer --> dm1[IonoDataManager<br/>dias/dataIO/data.py];
        dm1 --> model_sel{模型选择<br/>Dias_Unet / Dias_FPN};
        model_sel --> model_train[(TensorFlow 模型)];
        model_train --> logs[hist.npy + result/*.txt];
        model_train --> ckpt[result/*/models];
        model_train --> viz[ImgSaveDir PNG];
    end

    subgraph 测试流程
        tester --> dm2[IonoDataManager];
        tester --> load_model[tf.keras.models.load_model<br/>Test.ModelPath];
        dm2 --> load_model;
        load_model --> preds[模型输出];
        preds --> post[get_minH_maxF<br/>dataPostProcess.py];
        post --> metrics[MinH/MaxF 评估矩阵];
        preds --> test_viz[Test.ImgSaveDir PNG];
        metrics --> result_test[result/test/*.npy];
    end

    subgraph 评估流程
        evaluator --> dm3[IonoDataManager];
        evaluator --> load_model;
        dm3 --> evaluator;
        load_model --> evaluator;
        evaluator --> reports[统计结果 / 控制台输出];
    end

    ds --> dm1;
    ds --> dm2;
    ds --> dm3;

```