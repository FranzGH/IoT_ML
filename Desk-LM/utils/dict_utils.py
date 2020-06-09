
def parse_related_properties(property, dic, def_val):
    if property in dic:
        return [dic[property]]
    elif property+"_array" in dic:
        return dic[property+'_array']
    elif property+"_lowerlimit" in dic or property+"_lowerlimit" in dic:                    
        if property+"_lowerlimit" in dic:
            l = dic[property+'_lowerlimit']
        else:
            raise ValueError(error.errors['missing_lower_limit'] + ' for property' + property)
        if property+"_upperlimit" in dic:
            u = dic[property+'_upperlimit']
        else:
            raise ValueError(error.errors['missing_upper_limit'] + ' for property' + property)
        if property+"_interval" in dic:
            import numpy as np
            i = dic[property+'_interval']
            return np.arange(l, u, i)
        else:
            return np.arange(l, u)
    else:
        return [def_val]