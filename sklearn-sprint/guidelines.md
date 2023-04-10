
![Hugging Face x Scikit-learn](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hfxsklearn.png)

In this sprint, we will build interactive demos from the scikit-learn documentation and, afterwards, contribute the demos directly to the docs.

## Important Dates

🌅 Sprint Start Date: Apr 12, 2023
🌃 Sprint Finish Date: Apr 30, 2023

## To get started 🤩

1. Join our [Discord](https://huggingface.co/join/discord) and take the role #sklearn-sprint-participant by selecting "Sklearn Working Group" in the #role-assignment channel. Then, meet us in #sklearn-sprint channel.
2. Head to [this page](https://scikit-learn.org/stable/auto_examples/) and pick an example you’d like to build on. 
3. Leave a comment on [this spreadsheet](https://docs.google.com/spreadsheets/d/14EThtIyF4KfpU99Fm2EW3Rz9t6SSEqDyzV4jmw3fjyI/edit?usp=sharing) with your name under Owner column, claiming the example. The spreadsheet has a limited number of examples. Feel free to add yours with a comment if it doesn’t exist in the spreadsheet.
.
4. Start building!
    
    We will be hosting our applications in [scikit-learn](https://huggingface.co/sklearn-docs) organization of Hugging Face. 
    
    For complete starters: in the Hugging Face Hub, there are repositories for models, datasets, and [Spaces](https://huggingface.co/spaces). Spaces are a special type of repository hosting ML applications, such as showcasing a model. To write our apps, we will only be using Gradio. [Gradio](https://gradio.app/) is a library that lets you build a cool front-end application for your models, completely in Python, and supports many libraries! In this sprint, we will be using mostly visualization support (`matplotlib`, `plotly`, `altair` and more) and [skops](https://skops.readthedocs.io/en/stable/) integration (which you can launch an interface for a given classification or regression interface with one line of code). 
    
    In Gradio, there are two ways to create a demo. One is to use `Interface`, which is a very simple abstraction. Let’s see an example.
    
    ```python
    import gradio as gr
    
    # implement your classifier here 
    clf.fit(X_train, y_train)

    def cancer_classifier(df):
        # simply infer and return predictions
        predictions = clf.predict(df)
        return predictions
    
    gr.Interface(fn=cancer_classifier, inputs="dataframe", 
    outputs="label").launch()
    
    # save this in a file called app.py
    # then run it 
    ```
    
    This will result in following interface:
    
    ![Simple Interface](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/interface.png)
    
    This is very customizable. You can specify rows and columns, add a title and description, an example input, and more. There’s a more detailed guide [here](https://gradio.app/using-gradio-for-tabular-workflows/). 
    
    Another way of creating an application is to use [Blocks](https://gradio.app/quickstart/#blocks-more-flexibility-and-control). You can see usage of Blocks in the example applications linked in this guide. 
    
    After we create our application, we will create a Space. You can go to [hf.co](http://huggingface.co), click on your profile on top right and select “New Space”.
    
    ![New Space](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/new_space.png)
    
    We can name our Space, pick a license and select Space SDK as “Gradio”. Free hardware is enough for our app, so no need to change it.
    
    ![Space Configuration](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/space_config.png)
    
    After creating the Space, you have three options
     * You can clone the repository locally, add your files, and then push them to the Hub.
     * You can do all your coding directly in the browser.
     *  (shown below) You can do the coding locally and then drag and drop your application file to the Hub.
    
    ![Space Config](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/space_config.png)
    
    To upload your application file, pick “Add File” and drag and drop your file.
    
    ![New Space Landing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/space_landing.png)
    
    Lastly, if your application includes any library other than Gradio, create a file called requirements.txt and add requirements like below: 
    
    ```python
    matplotlib==3.6.3
    scikit-learn==1.2.1
    ```
    
     And your app should be up and running!
    
    **Example Submissions**
    
    We left couple of examples below: (there’s more at the end of this page)
    Documentation page for comparing linkage methods for hierarchical clustering and example Space built on it 👇🏼 
    
    [Comparing different hierarchical linkage methods on toy datasets](https://scikit-learn.org/stable/auto_examples/cluster/plot_linkage_comparison.html#sphx-glr-auto-examples-cluster-plot-linkage-comparison-py)
    
    [Hierarchical Clustering Linkage - a Hugging Face Space by scikit-learn](https://huggingface.co/spaces/scikit-learn/hierarchical-clustering-linkage)
    
    Note: If for your demo you're training a model from scratch (e.g. training an image classifier), you can push it to the Hub using [skops](https://skops.readthedocs.io/en/stable/) and build a Gradio demo on top of it.  For such submission, we expect a model repository with a model card and the model weight as well as a simple Space with the interface that receives input and outputs results. You can use this tutorial to get started with [skops](https://www.kdnuggets.com/2023/02/skops-new-library-improve-scikitlearn-production.html).
    
    You can find an example submission for a model repository below.
    
    [scikit-learn/cancer-prediction-trees · Hugging Face](https://huggingface.co/scikit-learn/cancer-prediction-trees)
    
4. After the demos are done, we will open pull requests to scikit-learn documentation in [scikit-learn’s repository](https://github.com/scikit-learn/scikit-learn) to contribute our application codes to be directly inside the documentation. We will help you out if this is your first open source contribution. 🤗 

**If you need any help** you can join our discord server, take collaborate role and join `sklearn-sprint` channel and ask questions 🤗🫂 

### Sprint Prizes
We will be giving following vouchers that can be spent at [Hugging Face Store](https://store.huggingface.co/) including shipping,
- $20 worth of voucher for everyone that builds three demos,
- $40 worth of voucher for everyone that builds five demos. 