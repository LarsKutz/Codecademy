# 13 Communicating Data Science Findings

<br>

## Content 
- **Communicating Data Science Findings**
    - [Article: Communicating Data Science Findings](#article-communicating-data-science-findings)
        - [Best Practices for Communicating Data Science Findings](#best-practices-for-communicating-data-science-findings)
            - [Know Your Audience](#know-your-audience)
            - [Structure The Sequence Of Your Story](#structure-the-sequence-of-your-story)
            - [Find and pick relevant data](#find-and-pick-relevant-data)
            - [Choose proper methods of visualization](#choose-proper-methods-of-visualization)
            - [Complexity and Purpose](#complexity-and-purpose)
            - [Conclusion: Communicating Data Science Findings](#conclusion-communicating-data-science-findings)
    - [Article: Structure of a Data Analysis Report](#article-structure-of-a-data-analysis-report)
        - [Structure](#structure)
        - [Target Audiences](#target-audiences)
            - [Primary Audience](#primary-audience)
            - [Secondary Audience](#secondary-audience)
        - [Key Features](#key-features)
        - [Detailed Outline](#detailed-outline)  
            - [1. Introduction](#1-introduction)
            - [2. Body](#2-body)
            - [3. Conclusion(s)/Discussion](#3-conclusionsdiscussion)
            - [4. Appendix/Appendices](#4-appendixappendices)
    - [Article: Audience Analysis - Just Who Are These Guys?](#article-audience-analysis---just-who-are-these-guys)
        - [Understanding Your Audience](#understanding-your-audience)
            - [Four Types of Audiences](#four-types-of-audiences)
        - [Adapting Your Writing to the Audience](#adapting-your-writing-to-the-audience)
        - [Situational Analysis](#situational-analysis)
        - [Tools for Audience and Situation Planning](#tools-for-audience-and-situation-planning)
    - [Article: How to Write Data Analysis reports. Lesson -2 know your audience](#article-how-to-write-data-analysis-reports-lesson--2-know-your-audience)
        - [1. Data Literacy ](#1-data-literacy)
        - [2. Subject Knowledge  ](#2-subject-knowledge)
        - [3. Time Constraints](#3-time-constraints)
        - [4. Not Everyone Is a Math Person](#4-not-everyone-is-a-math-person)
        - [5. Not Everyone Likes Charts](#5-not-everyone-likes-charts)

<br>

## Article: Communicating Data Science Findings
- Data Science is as much about storytelling as it is about coding and data wrangling. 
- The reason Data Scientists are in such high demand recently is because of their abilities to translate the performance and health of a company by gathering previously unavailable insights from its data. 
- It might seem like an easy and obvious thing at the moment, but in reality, a company’s data might be so jumbled and unstructured that mere mortals are unable to make any sense of it.

<br>

## Best Practices for Communicating Data Science Findings
- A Data Scientist explaining to a group of Product Managers that the covariance between two variables shows no visible linear relationship is like Dumbledore teaching a group of Jedi how to conjure up a Patronus. Two different worlds. Does not compute.
- The transformation of data analysis results by Data Scientists into easily accessible and understandable mediums for everyone to understand is actually one of the most important skills on the job.
- After a Data Scientist completes an in-depth investigation on some set of data, the results of this investigation need to be converted somehow into a simplified summary for others to observe and understand. 
- Usually, these kinds of summaries are written up as comments to one of your assigned tickets or more often as slides in a presentation that you then verbally discuss in further detail.

<br>

### Know Your Audience
- Before starting the process of communicating your findings, it’s important to put into perspective who your audience actually is. 
- The entire tone of your presentation is shaped and changed by how individuals will perceive it.
- Imagine you’re presenting information about tipping and restaurant billing transactions to a group of restaurant owners. 
- How would your presentation look? Now imagine you’re showing the same data to a group of developers who are creating a food delivery app. 
- How would your presentation be different for this audience than your first one?
- If you put too much math/statistics jargon into a report or presentation for non-technical people, they’re either going to ask you to define things for them or ask for you to clarify upon certain concepts. 
- Hand it over too simplified to a technical audience, however, and they’ll ask you how you came up with your results.
- You need to find the perfect blend between these two to correctly adapt your communication.

<br>

### Structure The Sequence Of Your Story
- A structure defines in what sequence you reveal the results of your analysis. 
- During this process, you also want to determine how much analytical depth you want to dive into. 
- This is an opportunity to highlight only the most important observations you derived from the data through your exploration of it.
- The type of observations you want to include is the kind that relates most to whatever the ultimate goal for the investigation is.
- If for example, you notice a completely unrelated trend in user behavior relative to your analysis, save it as an additional discussion point inside of an appendix at the end somewhere in your slides or report. 
- Sometimes an audience member might also pick up on this additional detail and ask you about it during a Q&A session. 
- This type of structure allows you to quickly go to your appendix and pick out a slide that goes more in-depth on a previously alluded detail.

<br>

### Find and pick relevant data
- Are all of the insights you gathered from your data absolutely necessary to discuss? 
- Does it help get your ultimate point across or is it just extra fluff you think is neat as a Data Scientist?
- Sometimes, additional statistics that help a Data Scientist to take a particular investigative path may not be the ideal insights to share with the audience. 
- If a number of features are vague in their immediate representation and don’t explain some kind of requested result, it’s better to pass on them.

<br>

- The following is an example of a visualization of the tipping and restaurant transaction data. 
- The visualization has had no adjustments and no added arguments. 
- On the X-axis we have the total bill amount, and on the Y-axis we have the tip amount. 
- The labels are small, there is no title, and the only immediate thing you can extract from it is the higher the bill, the higher the tip, which kind of seems obvious.
- This is a rookie mistake in the Data Science field when trying to communicate findings through visualizations. 
- We shouldn’t look at a chart and think *What is this chart’s purpose?*
    <img src="images/plain_scatter.webp" width="700">

<br>

### Choose proper methods of visualization
- Contrary to popular belief, your data usually has a finite set of visualization types you should probably apply to it. 
- Some examples include bar-charts and count-plots for categorical data, time-series charts for datetime type data, and lineplots/boxplots or histograms for continuous and numeric type data.
- **For example:**  
    <img src="images/bubble-graph.webp" width="200">
- What is this visualization trying to tell us?
- Using the wrong combination here will not only end up confusing the audience but could sacrifice the audiences’ trust in you and your ability to deliver reliable analyses. 
- For example, seeing a bubble plot type visualization on a website or in a magazine might look cool at first, but writing so much code to put circles behind 7 numbers is just silly. 
- The visualization above could have just been easily put into a table without any effort and much quicker to read.

<br>

### Complexity and Purpose
- Don’t let your visualizations become too noisy. 
- You don’t need to show a plot of every single possible combination of correlations for every single variable.
- Focus only on the features important within your dataset to prevent introducing way too much information at one time to your audience.
- This includes defining maximum thresholds for how many segmented or grouped trends to picture in one visualization.

<br>

- Another mistake made when trying to communicate a large magnitude of data is trying to fit it all in one visualization. 
- In cases where you’d like to plot multiple trends in your data, it may sometimes be a better idea to just split it up into separate charts.
- The following line plot shows the linear relation between bill amount and tips on different days. 
- There are actually only four days included in this dataset, but if we took out the legend and asked what you see, the only response we would expect to hear is *spaghetti*:
    <img src="images/spaghetti.webp" width="700">
- Since the trend for the tip amount seems to stay relatively similar on different days, there’s no real reason to visualize each day separately. 
- The only reason Friday looks like a more stable behavior is that there is less data. 
- If this dataset contained some kind of datetime variable and was much bigger, another approach would be to average out or sum the transactions by each date contained in the X-axis. 
- In this case, you would have to add additional ticks in the X-axis to help signify the days of the week but would look much cleaner.

<br>

### Conclusion: Communicating Data Science Findings
- Data always starts out messy, ugly, and confusing. 
- The simplest way of getting an audience to have a feel for a data-focused concept is through visualization. 
- When using visualization to present your findings, remember the following:
    - Visualizations are like jokes, if you don’t immediately understand them… they’re probably not very good.
    - Know who your audience is.
    - Make sure your visualization has a purpose and is somehow tied to the ultimate conclusion of your findings because irrelevant topics sometimes make your audience lose interest.
    - Make sure you are using relevant data.
    - Choose a proper method of visualization.
    - Use large, short labels on your graphs so that people can quickly understand what the markers or line in a chart actually mean.
    - Try to simplify concepts without using any technical jargon that makes the data look messy again.
- As you venture out of analytics and into other parts of the Data Science spectrum, there are going to be additional factors that come into play to best communicate your findings; however, no matter the area of Data Science you focus on, the practices mentioned here are always going to stay relevant.

<br>

## Article: Structure of a Data Analysis Report
- [Link to this Article](https://www.stat.cmu.edu/~brian/701/notes/paper-structure.pdf)

<br>

- A data analysis report differs from other types of professional writing. It is related to but distinct from:
    - A typical psych/social science paper with sections such as "Introduction/Methods/Analysis/Results/Discussion."
    - A research article in an academic journal.
    - An essay.
    - A lab report in a science class.

<br>

### Structure
- The overall structure of a data analysis report is simple:
    1. **Introduction**
    2. **Body**
    3. **Conclusion(s)/Discussion**
    4. **Appendix/Appendices**

<br>

### Target Audiences
- The data analysis report is written for several different audiences at the same time:

<br>

#### Primary Audience
- A primary collaborator or client who reads the Introduction and Conclusion to understand the work and may skim through the Body for additional details. 
- The report should be structured to facilitate a discussion with this audience.

<br>

#### Secondary Audience
- **Executive Person**: Likely to skim the Introduction and Conclusion.
- **Technical Supervisor**: Focuses on the Body and Appendix for quality control and statistical methodology verification.

<br>

### Key Features
- A data analysis report should: 
    - Be structured to allow different audiences to easily skim through it.
    - Maintain clear and unobtrusive writing to highlight the content rather than the prose.
- Common distractions to avoid:
    - Overly formal, flowery, or casual writing.
    - Grammatical and spelling errors.
    - Excessively broad or narrow contextual framing.
    - Overemphasis on process rather than outcomes.
    - Unnecessary technical details in the Body instead of the Appendix.
- The report serves as an internal communication tool while also informing supervisors or executives about the work conducted.

<br>

### Detailed Outline

#### 1. Introduction
- The introduction should include:
    - Summary of the study and data.
    - The key questions addressed by the analysis and summary conclusions.
    - Brief outline of the paper.

<br>

#### 2. Body
- The body can be structured in different ways:

**Traditional Structure**  
- Divides the Body into sections:
    - **Data**
    - **Methods**
    - **Analysis**
    - **Results**

**Question-Oriented Structure**
- Organized by research questions, with subsections such as:
    - Analysis
        - **Success Rate**
            - Methods
            - Analysis
            - Conclusions
        - **Time to Relapse**
            - Methods
            - Analysis
            - Conclusions
        - **Effect of Gender**
            - Methods
            - Analysis
            - Conclusions
        - **Hospital Effects**
            - Methods
            - Analysis
            - Conclusions
- Each section should contain tables or graphs for clarity, but excessive graphical material should be placed in the Appendix.

<br>

#### 3. Conclusion(s)/Discussion
- Summarizes the questions and conclusions, possibly adding observations, new questions, or future research directions.

<br>

#### 4. Appendix/Appendices
- Contains detailed and ancillary materials such as:
    - Technical descriptions of statistical methods.
    - Detailed tables or computer output.
    - Additional figures.
    - Computer code with comments.
- The Body should contain just enough information to make the point, while the Appendix provides additional details and references to specific sections.

<br>

## Article: Audience Analysis - Just Who Are These Guys?
- [Link to this Article](https://mcmassociates.io/textbook/aud.html)

<br>

## Understanding Your Audience  
- When writing technical documents, you must analyze your audience carefully. Your readers will have different levels of knowledge, expectations, and needs. 
- Understanding your audience ensures that your document is clear, relevant, and useful.  

### Four Types of Audiences  
1. **Experts**  
   - Have deep theoretical and specialized knowledge.  
   - Often involved in research and development.  
   - Expect detailed and precise technical content.  

2. **Technicians**  
   - Possess hands-on technical skills.  
   - Build, maintain, and repair products.  
   - Prefer clear, practical instructions over theory.  

3. **Managers (Executives, Administrators, Decision Makers)**  
   - Have limited technical knowledge.  
   - Need high-level overviews with cost-benefit analysis.  
   - Prefer summaries, visuals, and key takeaways.  

4. **General Readers (Non-Specialists, Laypersons)**  
   - Little to no technical background.  
   - Seek information out of curiosity or practical interest.  
   - Prefer simple explanations, everyday examples, and clear visuals.  

<br>

## Adapting Your Writing to the Audience  
- To communicate effectively, consider:  
    - **Background knowledge**: How familiar is your audience with the topic?  
    - **Purpose**: What does your audience need to know or do?  
    - **Cultural and linguistic differences**: How might these impact understanding?  

- To tailor your writing:  
    - Use **appropriate terminology** based on your audience’s knowledge.  
    - Adjust the **level of detail**—more for experts, less for general readers.  
    - Provide **visual aids**, examples, or summaries as needed.  

<br>

## Situational Analysis  
- Besides understanding your audience, consider the context in which your document will be used. Ask yourself:  
    - **What is the purpose?** Inform, instruct, persuade?  
    - **Who is the intended audience?**  
    - **What are the constraints?** Word limit, technical format, accessibility?  
    - **What is the tone and style?** Formal or informal? Concise or detailed?  

<br>

## Tools for Audience and Situation Planning  
- **Audience Planner**  
  - Helps define the target audience’s knowledge level, needs, and expectations.  

- **Situation Planner**  
  - Assists in adjusting the content based on purpose, setting, and audience.  

- By carefully analyzing both audience and context, technical writers can create documents that are precise, effective, and user-friendly.  

<br>

## Article: How to Write Data Analysis reports. Lesson -2 know your audience
- "*Consider the audience*" is one of the most common pieces of advice in storytelling and data visualization. 
- The general recommendation is to think about the audience’s needs, goals, and ability to understand the elements presented in the visualization.  
- This ensures that our message reaches them more effectively and resonates better. 
- However, the idea of considering the audience is somewhat vague.
- Some of us have a natural sense of empathy and quickly find the right approach, while others may struggle and need a more concrete checklist.  
- For example, we have guidelines for dealing with color blindness—an audience characteristic—but such examples usually stand alone.  
- In this article, I will break down the vague suggestion to "consider the audience" into something more checklist-like, helping you ensure the audience is truly considered.  

<br>

## 1. Data Literacy  
- Since data visualization is about presenting data, we can start by examining how familiar the audience is with data in general.  
- Highly data-literate individuals likely come from engineering, data science, or analytics backgrounds. 
- Surprisingly, some non-technical fields, like psychology, also have strong statistical components. 
- These individuals can understand complex statistical charts such as scatter plots or box plots and grasp advanced statistical concepts like confidence intervals and linear models.  
- Less data-literate individuals may still be highly educated but might come from fields like literature or visual arts. 
- They are less familiar with complex graphs, so simpler visualizations—such as bar charts, lines, and dots—are safer options.
- Avoid overwhelming them with statistical jargon; instead, explain a machine learning model as a friendly robot sipping warm oil.  

<br>

## 2. Subject Knowledge  
- Another key consideration is whether the audience is highly familiar with the subject or completely new to it.  
- Subject matter experts prefer to skip introductions and explanations. 
- They may not struggle with abbreviations and might even prefer them over full terms. 
- In some cases, there are established conventions for data visualization within a field, and deviating from them could be seen as overly creative or misplaced.  
- For example, if you’re presenting to finance professionals, you don’t need to explain EBITDA or WACC—or even spell them out.
- Finance experts may have a strong preference for displaying data in ways that mirror financial statements. 
- Waterfall charts, for example, are commonly used in finance but much less frequent in other fields. 
- Be careful—finance professionals can quickly spot minor inaccuracies!  
- On the other hand, people unfamiliar with the subject will need those introductions and abbreviations spelled out. 
- This means less time for insight analysis, so the key takeaway should be highlighted. 
- (If WACC increases, is that good or bad?) You may need to explain what "good" and "bad" mean in context and the consequences of each scenario—this could even become the primary focus of your visualization.  

<br>

## 3. Time Constraints  
- Time isn’t about the audience itself but rather the situation they’re in. 
- We often hear that C-level executives have short attention spans, but this could apply to other colleagues as well—especially if they’re skimming a chart while browsing a financial news article.
-  However, that same CEO might enjoy a deep read of their favorite economic journal after work.  
- When people have more time, the type of time also matters—is it leisure time, or is the meeting just incredibly long? 
- More time allows you to prepare the audience, ensuring they understand the topic, context, and situation. 
- It also enables you to use storytelling techniques—building suspense and emotion—especially in a structured presentation.  

<br>

## 4. Not Everyone Is a Math Person  
- If you’re presenting to colleagues, you may already know which individuals don’t naturally think in mathematical terms. 
- Some audiences may have more of these individuals, while others may be entirely composed of them.  
- Think of presenting to children—they might enjoy graphs and pictures, but math is just not their thing (unless they are young Hawkings).  
- Those who *are* math-oriented will be comfortable with basic concepts like graphs, numbers, and averages.  
- Those who *aren’t* will need a different approach. Instead of showing numbers, show proportions. 
- Add more annotations and explanations for them to read. Storytelling and analogies become much more important to convey ideas, so use sequences of images and illustrations.  

<br>

## 5. Not Everyone Likes Charts  
- Unfortunately, some people just aren’t good at interpreting charts.
- There is a clear divide—some people love charts, while others prefer to read text.  
- If your audience consists of *chart people*, use the best visualizations possible.  
- If your audience consists of *non-chart people*, consider explaining data in text format. 
- Instead of a pie chart showing "35% of users chose X," just write, "One in three users chose X." Bullet points can also work well.  

---

- By following these five points—data literacy, subject knowledge, time constraints, math familiarity, and visualization preference—you can tailor your presentation or data visualization more effectively to your audience.  
- Understanding your audience ensures your message is clear, engaging, and impactful.  

<br>

##