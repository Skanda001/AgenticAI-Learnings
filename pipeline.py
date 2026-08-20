from agents import build_readeragent , build_searchagent , writer_chain , critic_chain

def runpipeline(topic: str)->dict:
    state = {}

    #sEARCH agent

    print("\n"+"=" *50)
    print("Step 1 search agent is working")
    print("="*50)

    searchagent = build_searchagent()
    searchresult = searchagent.invoke({
        "messages":[("user",f"Find recent reliable and detailed information about the {topic}")]
    })
    state["searchresult"] = searchresult['messages'][-1].content
    print("\n search result",state['searchresult'])


#reader agent
    print("\n"+"=" *50)
    print("Step 1 search agent is wokieng")
    print("="*50)

    reader_agent = build_readeragent()
    reader_result = reader_agent.invoke({
        "messages": ["user",
            f"Based on the following search results about '{topic}', "
            f"pick the most relevant URL and scrape it for deeper content.\n\n"
            f"Search Results:\n{state['searchresult'][:800]}"
        ]
    })

    state['scraped_content'] = reader_result['messages'][-1].content

    print("\nscreape content\n",state['scraped_content'])

    #writer chain

    print("\n"+"=" *50)
    print("Step 3 Writer is drafing the report")
    print("="*50)

    research_combined = (
        f"SEARCH RESULTS:\n{state['searchresult']}\n\n"
        f"DETAILED SCRAPED CONTENT:\n{state['scraped_content']}"
    )

    state["report"] = writer_chain.invoke({
        "topic": topic,
        "research": research_combined
    })

    print("\n Final report\n",state['report'])

    #critic report

    print("\n"+"=" *50)
    print("Step 3 Writer is drafing the report")
    print("="*50)

    state["feedback"] = critic_chain.invoke({
        "report": state['report']
    })
    print("\n critic report\n",state['feedback'])

    return state

if __name__ =="__main__":
    topic = input("\nEnter a research topic : ")
    runpipeline(topic)