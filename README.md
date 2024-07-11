
<html>
<head>
    <title>My Website</title>
    <style>
         body {
            background-color: #FCFCF9; 
            margin: 0;
            padding: 0;
            align-items: center;
            justify-content: center;
            font-family: Arial, sans-serif;
        }
       .button-container {
            display: flex;
        }
        .button-container button {
            margin: 0;
            padding: 10px 20px;
            font-weight: 500;
            border: 1px solid #ccc;
            background-color: #C9C9C5;
            cursor: pointer;
        }
        .button-container button:hover {
            background-color: #969693;
        }
    </style>
</head>
<body>

<div class="button-container">
    <button onclick="scrollToSection('About me')">About me</button>
    <button onclick="scrollToSection('Technical skills')">Technical skills</button>
    <button onclick="scrollToSection('Projects')">Projects</button>
</div>

<section id="About me">
    <h2>👤 About Me </h2>
    <div style="display: flex; align-items: center;">
  <img src="images/profile_pic.png" alt="Alt text" style="margin-right: 10px; width: 100px;">
  <p> <br>
I am an NLP Data Science enthusiast with expertise in developing custom data-driven models across various domains,Fine-Tuning models using PEFT methods, leveraging large language model(LLM) capabilities with Langchain, creating multi-agent systems with Langchain for intelligent assistance, and building conversational chatbots.</p>
</div>

</section>

<section id="Technical skills">
    <h2>🛠️ Technical Skills </h2>
<p>
    <ul>
    <li> <strong>Programming Languages:</strong> Python, C++ </li>
 <li>   <strong>Data Preprocessing and Visualization:</strong> SQL, Pandas, Matplotlib, Seaborn </li>
<li> <strong>ML/DL Libraries and Frameworks:</strong> Keras, TensorFlow, Scikit Learn, Numpy </li>
<li> <strong>NLP Libraries:</strong> spaCy, NLTK, Gensim, Transformers, PEFT, Langchain, LlamaIndex </li>
<li> <strong>Versioning Tools:</strong> Git, mlflow, DVC </li>
<li> <strong>Others:</strong> Streamlit </li>
    </ul> </p> </section>

<section id="Projects">
    <h2>🚀 Projects</h2>

</section>

<script>
    function scrollToSection(id) {
        document.getElementById(id).scrollIntoView({ behavior: 'auto' });
    }
</script>

</body>
</html>


### Translator with Transformer Architecture [(Project link)](https://github.com/kalyan926/Translator)

- English to Telugu translator with complete **Transformer architecture** built from scratch using **keras subclassing APIs** for flexibility and control over the architecture's design and customization.
- Trained the model with custom training loop and scheduled learning rate for faster convergence. Used **kerastuner** library for best selection among the range of hyperparameter models using custom metric. The model's translation quality is evaluated using **BLEU** score metric.
- The Transformer architecture with multihead attention greatly improves the model's capability to manage long-term dependencies between words, offering improved contextual representation, capturing subtle nuances and semantic relationships crucial for translation and evaluated through **BLEU** score testing.

  
### Fine Tuning LLAMA2 Using QLORA [(Project link)](https://github.com/kalyan926/FineTuning-using-QLORA)

- Finetuning the base **LLAMA2-7b** using **QLORA** method with significantly fewer trainable parameters(<1%) to achieve faster training on limited hardware resources (single GPU). Finetuning is done with python code datasets for chat completions of coding related tasks.
- The base LLAMA2 model is quantized to 4 bit Normal Float using bitsandbytes library and then added the low rank adapter(**LORA**) layer to linear layers of model using **PEFT** library for fine tuning. Effectiveness of model is evaluated using **CODEBLEU** metric.
- Fine-tuning with the QLORA technique achieved 90% performance of the original full-precision trained model, using limited resources (a single GPU).

### 🧠 Multiagent AI Assistant [(Project link)](https://github.com/kalyan926/MultiAgent_AI_Assistant)

- Built a multi-agent system for intelligent task completion using **LANGCHAIN** and **OpenAI APIs** to leverage GPT-4 for improving productivity and efficiency of users.
- Implemented 4 specialized agents: **Task Planner**, **Task Executor**, **Evaluator** and **Feedback Agent** each designated for specific tasks and these agents are interconnected to form 
  a graph structure for facilitating seamless communication and task execution. 
- Agents are integrated with **Memory** for storing intermediate results and context, **RAG** to filter and provide only relevant information to the LLM's from the tools output, preventing **hallucinations**, **prompt injection** and **priming** and variety of **Tools** integration for enhancing variety of tasks completion accurately and efficiently.
- The project leverages LLM's reasoning and capabilities by **50%**, enhancing the multi-agents' ability to understand and complete complex tasks effectively.




