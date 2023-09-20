The FUNSD_polygon dataset has the following properties:

```json
{
   "form":[
      {
         "text":"12 /10 /98",
         "linking":[
            [
               2,
               27
            ]
         ],
         "label":"answer",
         "words":[
            {
               "text":"12",
               "polygon":[
                  184,
                  406,
                  198,
                  406,
                  198,
                  420,
                  184,
                  420
               ]
            },
            {
               "text":"/10",
               "polygon":[
                  198,
                  405,
                  216,
                  405,
                  216,
                  423,
                  198,
                  423
               ]
            },
            {
               "text":"/98",
               "polygon":[
                  215,
                  406,
                  233,
                  406,
                  233,
                  423,
                  215,
                  423
               ]
            }
         ],
         "id":27,
         "polygon":[
            184,
            405,
            233,
            405,
            233,
            423,
            184,
            423
         ]
      }
   ]
}
```
Where each property has the following meaning:
- `form`: The form that contains the text.
- `text`: The text of a given phrase.
- `linking`: The linking of an answer to a question. EG "DATE: 12/10/98"
- `label`: The label of the text. EG "answer"
- `words`: The words that compose the text.
- `id`: The id of the text.
- `polygon`: The polygon that contains the text in the format `[xleft, ytop, xright, ytop, xright, ybottom, xleft, ybottom]`
- `words.text`: The text of a given word.
- `words.polygon`: The polygon that contains the word in the format `[xleft, ytop, xright, ytop, xright, ybottom, xleft, ybottom]`

For more information on the dataset, please refer to the [original paper](https://arxiv.org/pdf/1905.13538.pdf).