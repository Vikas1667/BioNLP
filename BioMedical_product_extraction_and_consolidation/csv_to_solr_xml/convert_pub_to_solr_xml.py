from xml.sax.saxutils import escape


def convert_to_solrxml(article_dict):
    xml_string = '<doc>\n'
    f_close = '</field>\n'
    for each_key in article_dict:
        if isinstance(article_dict[each_key], list) and len(article_dict[each_key]) > 0:  # this is for list members
            for element in article_dict[each_key]: # over elements of list members
                if element is not None:
                    for each_inner_key in element:  # over the keys of inner dict
                        if element[each_inner_key] is not None:
                            if element[each_inner_key]:
                                current_line = '<field name="' + each_inner_key + '">' + escape(
                                    element[each_inner_key]) + f_close
                                xml_string = xml_string + current_line




        if (not isinstance(article_dict[each_key], list)) and article_dict[each_key] is not None:
            current_line = '<field name="' + each_key + '">' + escape(str(article_dict[each_key])) + f_close
            xml_string = xml_string + current_line

    xml_string = xml_string + '</doc>\n'
    return xml_string