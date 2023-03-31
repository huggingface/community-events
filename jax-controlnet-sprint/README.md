# JAX/Diffusers community sprint 🧨

Welcome to the JAX/Diffusers community sprint! The goal of this sprint is to work on fun and creative diffusion models using JAX and Diffusers.

In this event, we will create various applications with diffusion models in JAX/Flax and Diffusers using free TPU hours generously provided by Google Cloud.

This document is a walkthrough on all the important information to make a submission to the JAX/Diffusers community sprint.

Don't forget to fill out the [signup form]! 

> 💡 Note: This document is still WIP and it only contains initial details of the event. We will keep updating this document as we make other relevant information available throughout the community sprint.

## Table of Contents

- [Organization](#organization)
- [Important dates](#important-dates)
- [Communication](#communication)
- [Talks](#talks)

## Organization 

Participants can propose ideas for an interesting project involving Diffusion models. Teams of 3 to 5 will then be formed around the most promising and interesting projects. Make sure to read through the [Projects] (TODO) section on how to propose projects, comment on other participants' project ideas, and create a team.

To help each team successfully finish their project, we will organize talks by leading scientists and engineers from Google, Hugging Face, and the open-source diffusion community. The talks will take place on 17th of April. Make sure to attend the talks to get the most out of your participation! Check out the [Talks] (TODO) section to get an overview of the talks, including the speaker and the time of the talk.

Each team is then given **free access to a TPU v4-8 VM** from April 14 to May 1st. In addition, we will provide a training example in JAX/Flax and Diffusers to train [ControlNets](https://huggingface.co/blog/controlnet) to kick-start your project. We will also provide examples of how to prepare datasets for ControlNet training. During the sprint, we'll make sure to answer any questions you might have about JAX/Flax and Diffusers and help each team as much as possible to complete their projects!

At the end of the community sprint, each submission will be evaluated by a jury and the top-3 demos will be awarded a prize. Check out the [How to submit a demo] (TODO) section for more information and suggestions on how to submit your project.

> 💡 Note: Even though we provide an example for performing ControlNet training, participants can propose ideas that do not involve ControlNets at all. But the ideas need to be centered around Diffusion models.

## Important dates

- **29.03.** Official announcement of the community week. Make sure to fill out the [signup form].
- **31.03.** Start forming groups in #jax-diffusers-ideas channel in Discord. 
- **10.04.** Receiving acceptance e-mails, starting to collect data.
- **13.04. - 14.04. - [17.04.](https://www.youtube.com/watch?v=SOj2sxgvFe0)** Kick-off event with talks on Youtube. 
- **14.04. - 17.04.** Start providing access to TPUs. 
- **01.05.** Shutdown access to TPUs. 
- **08.05.**: Project presentations and determining the prize winners.

## Communication

All important communication will take place on our Discord server. Join the server using [this link](https://hf.co/join/discord). After you join the server, take the Diffusers role in `#role-assignment` channel and head to `#jax-diffusers-ideas` channel to share your idea as a forum post. To sign up for participation, fill out the [signup form] and we will give you access to two more Discord channels on discussions and technical support, and access to TPUs.
Important announcements of the Hugging Face, Flax/JAX, and Google Cloud team will be posted in the server.
The Discord server will be the central place for participants to post about their results, share their learning experiences, ask questions and get technical support in various obstacles they encounter.

For issues with Flax/JAX, Diffusers, Datasets or for questions that are specific to your project we will be interacting through public repositories and forums:

- Flax: [Issues](https://github.com/google/flax/issues), [Questions](https://github.com/google/flax/discussions)
- JAX: [Issues](https://github.com/google/jax/issues), [Questions](https://github.com/google/jax/discussions)
- 🤗 Diffusers: [Issues](https://github.com/huggingface/diffusers/issues), [Questions](https://discuss.huggingface.co/c/discussion-related-to-httpsgithubcomhuggingfacediffusers/63)
- 🤗 Datasets: [Issues](https://github.com/huggingface/datasets/issues), [Questions](https://discuss.huggingface.co/c/datasets/10)
- Project specific questions: Can be asked from each project's own post on #jax-diffusers-ideas channel on Discord.
- TPU related questions: #jax-diffusers-tpu-support channel on Discord. 
- General discussion: #jax-diffusers-sprint channel on Discord.
You will get access to #jax-diffusers-tpu-support and #jax-diffusers-sprint once you are accepted to attend the sprint.

When asking for help, we encourage you to post the link to [forum](https://discuss.huggingface.co) post to the Discord server, instead of directly posting issues or questions. 
This way, we make sure that the everybody in the community can benefit from your questions, even after the community sprint.

> 💡 Note: After 10th of April, if you have signed up on the google form, but you are not in the Discord channel, please leave a message on [the official forum announcement] (https://discuss.huggingface.co/t/controlling-stable-diffusion-with-jax-and-diffusers-using-v4-tpus/35187/2) and ping `@mervenoyan`, `@sayakpaul`, and `@patrickvonplaten`. We might take a day to process these requests.

## Talks

We will have talks from folks working at JAX & TPU teams at Google, diffusers and ethics teams at Hugging Face and awesome open-sorcerers working in generative AI. We will update this post with links to the talks, so keep an eye here or on Discord in diffusion models core-announcements channel and set your reminders!

- [Link to the talks on 17th of April](https://www.youtube.com/watch?v=SOj2sxgvFe0), we will be hosting Andreas Steiner, Margaret Mitchell and Boris Dayma. 

| Speaker	| Topic	| Time	| Video |
|---|---|---|---|
| Andreas Steiner, Google Brain	| JAX & ControlNet | 6.00pm-6.45pm CEST / 9.00am-9.45am PST| [![Youtube](https://www.youtube.com/s/desktop/f506bd45/img/favicon_32.png)](https://www.youtube.com/watch?v=SOj2sxgvFe0) |
| Boris Dayma, craiyon	| DALL-E Mini	| 6.45pm-7.30pm CEST / 9.45am-10.30am PST	| [![Youtube](https://www.youtube.com/s/desktop/f506bd45/img/favicon_32.png)](https://www.youtube.com/watch?v=SOj2sxgvFe0) |
|Margaret Mitchell, Hugging Face	| Ethics of Text-to-Image |	7.30pm-8.00pm CEST / 10.30am-11.00am PST	| [![Youtube](https://www.youtube.com/s/desktop/f506bd45/img/favicon_32.png)](https://www.youtube.com/watch?v=SOj2sxgvFe0) |

[signup form]: https://forms.gle/t3M7aNPuLL9V1sfa9
