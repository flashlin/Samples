using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class AutoSettleLeagueListEntityConfiguration : IEntityTypeConfiguration<AutoSettleLeagueListEntity>
    {
        public void Configure(EntityTypeBuilder<AutoSettleLeagueListEntity> builder)
        {
            builder.ToTable("AutoSettleLeagueList");

            builder.HasKey(x => x.leagueid);

            builder.Property(x => x.leagueid)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.leaguename)
                .HasColumnType("nvarchar(300)")
                .IsRequired()
                .HasMaxLength(300)
            ;

        }
    }
}
