-- Convert Pandoc Markdown tables into two-column-compatible LaTeX floats.
local function escape(s)
  return s:gsub('\\','\\textbackslash{}'):gsub('([%%#&_$])','\\%1')
end
local function textCell(c)
  return escape(pandoc.utils.stringify(pandoc.Pandoc(c.contents))):gsub('\n',' ')
end
local function rows(t)
  local out={}
  for _,r in ipairs(t.head.rows) do out[#out+1]=r end
  for _,b in ipairs(t.bodies) do
    for _,r in ipairs(b.body) do out[#out+1]=r end
  end
  for _,r in ipairs(t.foot.rows) do out[#out+1]=r end
  return out
end
local function tableFloat(t, caption, id)
  local rr=rows(t)
  local n=#t.colspecs
  local fmt
  if id==7 then
    fmt='>{\\raggedright\\arraybackslash}p{0.48\\columnwidth}r'
  elseif id==8 then
    fmt='>{\\raggedright\\arraybackslash}p{0.35\\columnwidth}rr'
  elseif id==2 then
    fmt='>{\\raggedright\\arraybackslash}p{0.28\\columnwidth}r>{\\raggedright\\arraybackslash}p{0.39\\columnwidth}'
  elseif id==9 then
    fmt='>{\\raggedright\\arraybackslash}p{0.73\\columnwidth}r'
  elseif id==10 then
    fmt='>{\\raggedright\\arraybackslash}p{0.44\\columnwidth}>{\\raggedright\\arraybackslash}p{0.46\\columnwidth}'
  elseif id==11 then
    fmt='>{\\raggedright\\arraybackslash}p{0.66\\columnwidth}rr'
  elseif n==3 then
    fmt='lrr'
  elseif n==4 then
    fmt='lrrr'
  elseif n==2 then
    fmt='lr'
  else
    fmt='l'..string.rep('r',n-1)
  end
  local out={'\\begin{table}[!htbp]', '\\centering'}
  if id>0 then
    out[#out+1]='\\caption{'..escape(caption)..'}'
    out[#out+1]='\\label{tab:waste-'..id..'}'
  end
  out[#out+1]='\\footnotesize'
  out[#out+1]='\\setlength{\\tabcolsep}{2.5pt}'
  out[#out+1]='\\renewcommand{\\arraystretch}{1.06}'
  out[#out+1]='\\begin{tabular}{'..fmt..'}'
  out[#out+1]='\\toprule'
  for i,r in ipairs(rr) do
    local cells={}
    for _,c in ipairs(r.cells) do cells[#cells+1]=textCell(c) end
    while #cells<n do cells[#cells+1]='' end
    if id==8 and i>1 and #cells>=3 then
      local p,r,f=cells[3]:match('([%d%.]+)%s*/%s*([%d%.]+)%s*/%s*([%d%.]+)')
      if p then cells[3]='\\shortstack[r]{'..p..'\\\\'..r..'\\\\'..f..'}' end
    end
    out[#out+1]=table.concat(cells,' & ')..' \\\\'
    if i==#t.head.rows and i<#rr then out[#out+1]='\\midrule' end
  end
  out[#out+1]='\\bottomrule'
  out[#out+1]='\\end{tabular}'
  out[#out+1]='\\end{table}'
  return pandoc.RawBlock('latex',table.concat(out,'\n'))
end
local function imageFloat(im)
  local cap=escape(pandoc.utils.stringify(im.caption))
  return pandoc.RawBlock('latex',
    '\\begin{figure*}[!t]\n\\centering\n'..
    '\\includegraphics[width=0.88\\textwidth,keepaspectratio]{'..im.src..'}\n'..
    '\\caption{'..cap..'}\n\\end{figure*}')
end
function Pandoc(doc)
  local result={}
  local i=1
  while i<=#doc.blocks do
    local b=doc.blocks[i]
    local n=doc.blocks[i+1]
    if (b.t=='Para' or b.t=='Plain') and n and n.t=='Table' then
      local id,caption=pandoc.utils.stringify(b):match('^Table%s+(%d+)%.%s*(.+)')
      if id then
        result[#result+1]=tableFloat(n,caption,tonumber(id))
        i=i+2
      else
        result[#result+1]=b;i=i+1
      end
    elseif b.t=='Table' then
      result[#result+1]=tableFloat(b,'',0)
      i=i+1
    elseif b.t=='Figure' then
      local im=nil
      for _,block in ipairs(b.content) do
        if block.t=='Plain' or block.t=='Para' then
          for _,inline in ipairs(block.content) do
            if inline.t=='Image' then im=inline; break end
          end
        end
        if im then break end
      end
      if im then
        result[#result+1]=imageFloat(im)
      else
        result[#result+1]=b
      end
      i=i+1
    elseif b.t=='Para' and #b.content==1 and b.content[1].t=='Image' then
      result[#result+1]=imageFloat(b.content[1]);i=i+1
    else
      result[#result+1]=b;i=i+1
    end
  end
  doc.blocks=result
  return doc
end